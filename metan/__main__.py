"""One entry point for configuration-driven METAN reproduction."""
import importlib
import sys

COMMANDS={
    'rebuild':'metan.rebuild',
    'reproduce':'metan.reproduce',
    'train':'metan.train',
    'cv':'metan.run_cv',
    'predict':'metan.predict',
    'baselines':'metan.baselines',
    'prepare':'metan.prepare',
    'check':'metan.check',
}

def main():
    if len(sys.argv)<2 or sys.argv[1] in ('-h','--help'):
        print('METAN commands: '+', '.join(COMMANDS))
        print('Run python -m metan <command> --help for arguments.')
        return
    command=sys.argv.pop(1)
    if command not in COMMANDS:
        raise SystemExit('Unknown command: '+command)
    importlib.import_module(COMMANDS[command]).main()

if __name__=='__main__':
    main()
