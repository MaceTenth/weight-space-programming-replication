#!/bin/zsh
# Creates <sandbox_dir>/py : runs python -I under sandbox-exec, read access only to system paths, the venv and the sandbox dir; no network.
dir=$(cd $1 && pwd); venv=/Users/shimonmoyal/weights/.venv
cat > $dir/py <<PY
#!/bin/zsh
exec sandbox-exec -p '(version 1)(deny default)(allow process-exec* process-fork)(allow sysctl-read)(allow mach-lookup)(allow ipc-posix-shm)(allow file-read-metadata)(allow file-read* (subpath "/usr") (subpath "/System") (subpath "/Library") (subpath "/private/etc") (subpath "/private/var/db") (subpath "/dev") (subpath "/opt") (subpath "$venv") (subpath "$dir") (literal "/"))(allow file-write* (subpath "$dir") (subpath "/private/tmp") (subpath "/dev"))(deny network*)' $venv/bin/python -I "\$@"
PY
chmod +x $dir/py; echo "wrote $dir/py"
