"""MiniWeb: a tiny deterministic text-based website for agent experiments.

A site is a set of pages. Each page shows a title, some text lines and a numbered
list of interactive elements (links, buttons, text inputs). The agent acts with
one of:

    CLICK(<id>)
    TYPE(<id>, "<text>")
    BACK
    DONE("<answer>")

Tasks are generated from templates with random (seeded) content so that train and
test sets share structure but not surface strings. Every task has an oracle policy
that solves it in the minimum number of steps; the oracle is used to build the
fine-tuning data and to define the step cap.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# ------------------------------------------------------------------ content pools
PRODUCTS = {
    "train": ["Walnut Desk Lamp", "Blue Rain Jacket", "Steel Water Bottle", "Cedar Picture Frame",
              "Cotton Beach Towel", "Bamboo Cutting Board", "Linen Throw Pillow", "Wool Hiking Socks",
              "Ceramic Coffee Mug", "Canvas Tote Bag", "Oak Bookshelf", "Copper Watering Can"],
    "test": ["Maple Serving Tray", "Green Garden Hose", "Glass Teapot", "Leather Notebook",
             "Marble Cheese Board", "Felt Laptop Sleeve", "Brass Door Knocker", "Silk Scarf"],
}
USERS = {
    "train": ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi"],
    "test": ["ivan", "judy", "mallory", "oscar", "peggy", "trent"],
}
QUERIES = {
    "train": ["lamp", "jacket", "bottle", "frame", "towel", "board", "pillow", "socks"],
    "test": ["tray", "hose", "teapot", "notebook", "scarf", "sleeve"],
}
SETTINGS = ["Email notifications", "SMS alerts", "Weekly newsletter", "Dark mode", "Two-factor login"]
CITIES = ["Boston", "Denver", "Austin", "Portland", "Madison", "Raleigh", "Tucson", "Omaha"]


# ------------------------------------------------------------------ site model
@dataclass
class Element:
    kind: str          # link | button | input
    label: str
    target: str | None = None   # page name for links / buttons
    name: str | None = None     # state key for inputs / toggles


@dataclass
class Page:
    title: str
    lines: list[str] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)


@dataclass
class Task:
    family: str
    split: str
    seed: int
    goal: str
    answer: str | None       # expected DONE answer (None = any)
    site: dict[str, Page]
    oracle: list[str]        # action strings
    check_state: dict = field(default_factory=dict)   # required state at DONE


def build_site(rng: random.Random, split: str) -> tuple[dict[str, Page], dict]:
    """Build a small shop site with randomised content. Returns (pages, facts)."""
    prods = rng.sample(PRODUCTS[split], 4)
    prices = {p: f"${rng.randint(8, 199)}.{rng.choice(['00', '50', '99'])}" for p in prods}
    phone = f"({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
    city = rng.choice(CITIES)
    user = rng.choice(USERS[split])
    pw = f"{rng.choice(['sun', 'moon', 'star', 'rock'])}{rng.randint(10, 99)}"
    settings_on = {s: True for s in SETTINGS}
    facts = dict(products=prods, prices=prices, phone=phone, city=city, user=user, pw=pw,
                 settings=settings_on)

    pages = {}
    pages["home"] = Page("Northwind Goods - Home",
                         ["Welcome to Northwind Goods, your neighbourhood shop."],
                         [Element("link", "Products", "products"),
                          Element("link", "Search", "search"),
                          Element("link", "Account", "account"),
                          Element("link", "Contact", "contact"),
                          Element("link", "Cart", "cart")])
    pages["products"] = Page("Products", ["All items in stock."],
                             [Element("link", p, f"product:{p}") for p in prods] +
                             [Element("link", "Home", "home")])
    for p in prods:
        pages[f"product:{p}"] = Page(p, [f"{p}. Price: {prices[p]}. Ships in 2 days."],
                                     [Element("button", "Add to cart", f"addcart:{p}"),
                                      Element("link", "Products", "products"),
                                      Element("link", "Home", "home")])
    pages["search"] = Page("Search", ["Type a word and press Search."],
                           [Element("input", "Search box", name="query"),
                            Element("button", "Search", "search_results"),
                            Element("link", "Home", "home")])
    pages["search_results"] = Page("Search results", [], [Element("link", "Home", "home")])
    pages["account"] = Page("Account", ["You are not logged in."],
                            [Element("link", "Log in", "login"),
                             Element("link", "Settings", "settings"),
                             Element("link", "Home", "home")])
    pages["login"] = Page("Log in", ["Enter your details."],
                          [Element("input", "Username", name="username"),
                           Element("input", "Password", name="password"),
                           Element("button", "Log in", "do_login"),
                           Element("link", "Home", "home")])
    pages["login_ok"] = Page("Logged in", [f"Welcome back, {user}!"],
                             [Element("link", "Account", "account"), Element("link", "Home", "home")])
    pages["login_fail"] = Page("Log in failed", ["Wrong username or password."],
                               [Element("link", "Try again", "login"), Element("link", "Home", "home")])
    pages["settings"] = Page("Settings", ["Toggle a setting by clicking it."],
                             [Element("button", s, f"toggle:{s}", name=s) for s in SETTINGS] +
                             [Element("link", "Account", "account"), Element("link", "Home", "home")])
    pages["contact"] = Page("Contact us", [f"Northwind Goods, {city}.", f"Phone: {phone}.",
                                           "Open Monday to Saturday."],
                            [Element("link", "Home", "home")])
    pages["cart"] = Page("Cart", [], [Element("button", "Checkout", "checkout"),
                                      Element("link", "Products", "products"),
                                      Element("link", "Home", "home")])
    pages["checkout"] = Page("Checkout", ["Review your order."],
                             [Element("button", "Place order", "order_ok"),
                              Element("link", "Cart", "cart")])
    pages["order_ok"] = Page("Order placed", ["Thank you! Your order is confirmed."],
                             [Element("link", "Home", "home")])
    return pages, facts


# ------------------------------------------------------------------ environment
ACTION_RE = re.compile(
    r"""(?P<click>CLICK\s*\(\s*(?P<cid>\d+)\s*\))|"""
    r"""(?P<type>TYPE\s*\(\s*(?P<tid>\d+)\s*,\s*["'“](?P<text>[^"'”]*)["'”]\s*\))|"""
    r"""(?P<back>\bBACK\b)|"""
    r"""(?P<done>DONE\s*\(\s*(?:["'“](?P<ans>[^"'”]*)["'”]|(?P<ans2>[^)]*))\s*\))""",
    re.IGNORECASE)


def parse_action(text: str):
    """Return (kind, arg) for the first action in text, or None."""
    m = ACTION_RE.search(text)
    if not m:
        return None
    if m.group("click"):
        return ("CLICK", int(m.group("cid")))
    if m.group("type"):
        return ("TYPE", (int(m.group("tid")), m.group("text")))
    if m.group("back"):
        return ("BACK", None)
    return ("DONE", (m.group("ans") if m.group("ans") is not None else m.group("ans2") or "").strip())


def canon(a) -> str:
    kind, arg = a
    if kind == "CLICK":
        return f"CLICK({arg})"
    if kind == "TYPE":
        return f'TYPE({arg[0]}, "{arg[1]}")'
    if kind == "BACK":
        return "BACK"
    return f'DONE("{arg}")'


class MiniWeb:
    STEP_CAP = 12

    def __init__(self, task: Task):
        self.task = task
        self.pages = task.site
        self.facts_prods = [e.label for e in self.pages["products"].elements if e.kind == "link" and e.target.startswith("product:")]
        self.reset()

    def reset(self):
        self.page = "home"
        self.history = ["home"]
        self.inputs: dict[str, str] = {}
        self.state = dict(logged_in=False, cart=[], order_placed=False,
                          settings={s: True for s in SETTINGS}, last_query=None)
        self.steps = 0
        self.msg = ""
        self.done = False
        return self.observe()

    # ---- rendering
    def _elements(self) -> list[Element]:
        p = self.pages[self.page]
        if self.page == "search_results":
            q = (self.state["last_query"] or "").lower()
            hits = [pr for pr in self.facts_prods if q and q in pr.lower()]
            els = [Element("link", pr, f"product:{pr}") for pr in hits] + list(p.elements)
            return els
        if self.page == "cart":
            return list(p.elements)
        return list(p.elements)

    def _lines(self) -> list[str]:
        p = self.pages[self.page]
        if self.page == "search_results":
            q = self.state["last_query"] or ""
            n = len([e for e in self._elements() if e.target and e.target.startswith("product:")])
            return [f'Results for "{q}": {n} item(s).'] if q else ["No search yet."]
        if self.page == "cart":
            items = self.state["cart"]
            return [f"Items in cart: {', '.join(items) if items else 'none'}."]
        if self.page == "settings":
            return ["Toggle a setting by clicking it."]
        if self.page == "account" and self.state["logged_in"]:
            return ["You are logged in."]
        return list(p.lines)

    def observe(self) -> str:
        out = [f"PAGE: {self.pages[self.page].title}"]
        out += self._lines()
        for i, e in enumerate(self._elements(), 1):
            if e.kind == "input":
                val = self.inputs.get(e.name, "")
                out.append(f'[{i}] input "{e.label}" (value: "{val}")')
            elif e.kind == "button" and e.name in self.state["settings"]:
                out.append(f'[{i}] button "{e.label}: {"ON" if self.state["settings"][e.name] else "OFF"}"')
            else:
                out.append(f'[{i}] {e.kind} "{e.label}"')
        if self.msg:
            out.append(f"NOTE: {self.msg}")
            self.msg = ""
        return "\n".join(out)

    # ---- dynamics
    def step(self, action):
        """action: (kind, arg) or None. Returns (obs, terminal_reason or None)."""
        self.steps += 1
        if action is None:
            self.msg = "Could not read an action. Use CLICK(n), TYPE(n, \"text\"), BACK or DONE(\"answer\")."
            return self.observe(), self._cap()
        kind, arg = action
        els = self._elements()
        if kind == "DONE":
            self.done = True
            return self.observe(), "done"
        if kind == "BACK":
            if len(self.history) > 1:
                self.history.pop()
                self.page = self.history[-1]
            return self.observe(), self._cap()
        idx = arg if kind == "CLICK" else arg[0]
        if not (1 <= idx <= len(els)):
            self.msg = f"There is no element [{idx}] on this page."
            return self.observe(), self._cap()
        e = els[idx - 1]
        if kind == "TYPE":
            if e.kind != "input":
                self.msg = f"[{idx}] is not a text input."
            else:
                self.inputs[e.name] = arg[1]
            return self.observe(), self._cap()
        # CLICK
        if e.kind == "input":
            self.msg = f"[{idx}] is a text input; use TYPE."
            return self.observe(), self._cap()
        t = e.target
        if t.startswith("addcart:"):
            self.state["cart"].append(t.split(":", 1)[1])
            self.msg = f"Added {t.split(':', 1)[1]} to cart."
        elif t.startswith("toggle:"):
            s = t.split(":", 1)[1]
            self.state["settings"][s] = not self.state["settings"][s]
        elif t == "do_login":
            u, p = self.inputs.get("username", ""), self.inputs.get("password", "")
            ok = (u == self.task.check_state.get("_user")) and (p == self.task.check_state.get("_pw"))
            self.state["logged_in"] = bool(ok)
            self._goto("login_ok" if ok else "login_fail")
        elif t == "search_results":
            self.state["last_query"] = self.inputs.get("query", "")
            self._goto("search_results")
        elif t == "checkout":
            if not self.state["cart"]:
                self.msg = "Your cart is empty."
            else:
                self._goto("checkout")
        elif t == "order_ok":
            self.state["order_placed"] = True
            self._goto("order_ok")
        else:
            self._goto(t)
        return self.observe(), self._cap()

    def _goto(self, page):
        self.page = page
        self.history.append(page)

    def _cap(self):
        return "step_cap" if self.steps >= self.STEP_CAP else None

    # ---- grading
    def success(self, answer: str | None) -> bool:
        t = self.task
        for k, v in t.check_state.items():
            if k.startswith("_"):
                continue
            if k == "setting_off":
                if self.state["settings"][v]:
                    return False
            elif k == "logged_in":
                if not self.state["logged_in"]:
                    return False
            elif k == "order_placed":
                if not self.state["order_placed"]:
                    return False
            elif k == "cart_has":
                if v not in self.state["cart"]:
                    return False
        if t.answer is not None:
            if answer is None:
                return False
            return _norm(t.answer) in _norm(answer)
        return True


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9$.]", "", s.lower())


# ------------------------------------------------------------------ tasks
def make_task(family: str, split: str, seed: int) -> Task:
    rng = random.Random(f"{family}-{split}-{seed}")
    pages, f = build_site(rng, split)
    prods = f["products"]
    check = {"_user": f["user"], "_pw": f["pw"]}
    if family == "price":
        p = rng.choice(prods)
        i = prods.index(p) + 1
        return Task(family, split, seed, f"What is the price of the {p}? Reply with DONE and the price.",
                    f["prices"][p], pages, ["CLICK(1)", f"CLICK({i})", f'DONE("{f["prices"][p]}")'], check)
    if family == "contact":
        return Task(family, split, seed, "Find the shop's phone number on the Contact page. Reply with DONE and the number.",
                    f["phone"], pages, ["CLICK(4)", f'DONE("{f["phone"]}")'], check)
    if family == "login":
        check = dict(check, logged_in=True)
        return Task(family, split, seed, f'Log in with username "{f["user"]}" and password "{f["pw"]}", then reply DONE.',
                    None, pages, ["CLICK(3)", "CLICK(1)", f'TYPE(1, "{f["user"]}")', f'TYPE(2, "{f["pw"]}")',
                                  "CLICK(3)", 'DONE("logged in")'], check)
    if family == "toggle":
        s = rng.choice(SETTINGS)
        i = SETTINGS.index(s) + 1
        check = dict(check, setting_off=s)
        return Task(family, split, seed, f'Turn off the "{s}" setting in Account > Settings, then reply DONE.',
                    None, pages, ["CLICK(3)", "CLICK(2)", f"CLICK({i})", 'DONE("turned off")'], check)
    if family == "search":
        q = rng.choice(QUERIES[split])
        hits = [p for p in prods if q in p.lower()]
        if not hits:   # make sure the query hits one product
            p = rng.choice(prods)
            q = p.split()[-1].lower()
            hits = [p]
        return Task(family, split, seed, f'Use the Search page to search for "{q}". Reply with DONE and the title of the first result.',
                    hits[0], pages, ["CLICK(2)", f'TYPE(1, "{q}")', "CLICK(2)", f'DONE("{hits[0]}")'], check)
    if family == "order":
        p = rng.choice(prods)
        i = prods.index(p) + 1
        check = dict(check, order_placed=True, cart_has=p)
        return Task(family, split, seed, f"Add the {p} to the cart and place the order, then reply DONE.",
                    None, pages, ["CLICK(1)", f"CLICK({i})", "CLICK(1)", "BACK", "BACK", "CLICK(5)", "CLICK(1)",
                                  "CLICK(1)", 'DONE("order placed")'], check)
    raise ValueError(family)


FAMILIES = ["price", "contact", "login", "toggle", "search", "order"]


def task_set(split: str, n_per_family: int, seed0: int = 0) -> list[Task]:
    return [make_task(fam, split, seed0 + k) for fam in FAMILIES for k in range(n_per_family)]


def verify_oracles(tasks):
    """Run every oracle; raise if any fails. Returns max oracle length."""
    worst = 0
    for t in tasks:
        env = MiniWeb(t)
        ans = None
        for a in t.oracle:
            act = parse_action(a)
            if act[0] == "DONE":
                ans = act[1]
            env.step(act)
        assert env.success(ans), (t.family, t.seed, t.goal, env.observe())
        worst = max(worst, len(t.oracle))
    return worst


if __name__ == "__main__":
    tr = task_set("train", 5)
    te = task_set("test", 5)
    print("train", len(tr), "test", len(te), "max oracle len", verify_oracles(tr + te))
    env = MiniWeb(te[0])
    print(te[0].goal)
    print(env.observe())
