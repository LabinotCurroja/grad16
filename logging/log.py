


colors = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "purple": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "end": "\033[0m"
}

class Logger:
    def __init__(self, name="GradLogger", debug=False):
        self.name  = name
        self.debug = debug 

    def info(self, message, color='blue'):
        print(f"{colors[color]}{self.name}::INFO: {message}{colors['end']}")

    def FATAL(self, message):
        print(f"{colors['red']}{self.name}::ERRO: {message}{colors['end']}")

    def warn(self, message):
        print(f"{colors['yellow']}[WARN] {self.name}: {message}{colors['end']}")

    def debug(self, message):
        if self.debug:
            print(f"{colors['purple']}[DEBUG] {self.name}: {message}{colors['end']}")
