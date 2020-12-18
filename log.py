import logging

log_format = '%(asctime)s %(message)s'
logging.basicConfig(filename=os.path.join(args.log_dir, f'seed_{args.seed}_log.txt'),
                    level=logging.INFO,
                    format=log_format,
                    datefmt='%m/%d %I:%M:%S %p')
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(sh)
