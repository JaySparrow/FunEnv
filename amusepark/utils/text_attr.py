class Attr:
    RESET = "\033[0m"

    # cancel SGR codes if we don't write to a terminal
    if not __import__("sys").stdout.isatty():
        for _ in dir():
            if isinstance(_, str) and _[0] != "_":
                locals()[_] = ""
    else:
        # set Windows console in VT mode
        if __import__("platform").system() == "Windows":
            kernel32 = __import__("ctypes").windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            del kernel32

class Foreground(Attr):
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[0;90m"
    LIGHT_RED = "\033[0;91m"
    LIGHT_GREEN = "\033[0;92m"
    YELLOW = "\033[0;93m"
    LIGHT_BLUE = "\033[0;94m"
    LIGHT_PURPLE = "\033[0;95m"
    LIGHT_CYAN = "\033[0;96m"
    LIGHT_WHITE = "\033[0;97m"

class Background(Attr):
    """ ANSI color codes """
    BLACK = "\033[0;40m"
    RED = "\033[0;41m"
    GREEN = "\033[0;42m"
    BROWN = "\033[0;43m"
    BLUE = "\033[0;44m"
    PURPLE = "\033[0;45m"
    CYAN = "\033[0;46m"
    LIGHT_GRAY = "\033[0;47m"
    DARK_GRAY = "\033[0;100m"
    LIGHT_RED = "\033[0;101m"
    LIGHT_GREEN = "\033[0;102m"
    YELLOW = "\033[0;103m"
    LIGHT_BLUE = "\033[0;104m"
    LIGHT_PURPLE = "\033[0;105m"
    LIGHT_CYAN = "\033[0;106m"
    LIGHT_WHITE = "\033[0;107m"

class Font(Attr):
    """ ANSI font codes """
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"