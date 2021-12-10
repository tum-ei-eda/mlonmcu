def makeDirName(base, *args, flags=None):
    # Creates a directory name based on configuration values.
    names = [base] + [name for check, name in args if check]
    if flags:
        names = names + flags
    return "_".join(names)

def makeFlags(*args):
    flags = []
    for cond, name in args:
        if cond:
            flags.append(name)
    return flags

# Executes a process with the given args and using the given kwards as Popen arguments.
def exec(*args, **kwargs):
    logger.info("- Executing: " + str(args))
    subprocess.run([i for i in args], **kwargs, check=True)


def exec_getout(*args, live=False, **kwargs):
    logger.info("- Executing: " + str(args))
    outStr = ""
    if live:
        process = subprocess.Popen([i for i in args], **kwargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            new_line = line.decode(errors="replace")
            outStr = outStr + new_line
            print(new_line.replace("\n", ""))
        assert process.poll() == 0, "The process returned an non-zero exit code! (CMD: `{}`)".format(" ".join(args))
    else:
        try:
            p = subprocess.run(
                [i for i in args], **kwargs, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            outStr = p.stdout.decode(errors="replace")
            logger.info(outStr)
        except subprocess.CalledProcessError as e:
            outStr = e.output.decode(errors="replace")
            logger.error(outStr)
            raise

    return outStr


# Makes sure all directories at the given path are created.
def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Clones a git repository at given url into given dir and switches to the given branch.
def clone(dir, url, branch="", rename=""):
    mkdirs(dir)
    try:
        if branch != "":
            if rename == "":
                exec("git", "clone", "--recursive", "-b", branch, url, cwd=dir)
            else:
                exec("git", "clone", "--recursive", "-b", branch, url, os.path.join(dir, rename))
        else:
            if rename == "":
                exec("git", "clone", "--recursive", url, cwd=dir)
            else:
                exec("git", "clone", "--recursive", url, os.path.join(dir, rename))
    except:
        logger.debug(str(url) + " is already there")

