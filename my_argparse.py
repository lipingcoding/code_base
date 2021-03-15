from argparse import ArgumentParser, HelpFormatter, Namespace
import yaml

class MyArgumentParser(ArgumentParser):
    """
    基于 argparse.ArgumentParser, 实现了将 yaml config 文件和命令行参数同步解析的功能.
    (1) parser 自动添加了 --config 参数, 用于提供 config 文件的路径 
    (2) 当通过命令行参数提供了参数值后, config 中定义的参数默认值会被覆盖
    (3) config 文件采用 YAML 格式(需要安装 PyYAML 包), 格式如 "key: value"
    (4) 用法与 ArgumentParser 完全相同, 返回类型为 Namespace 的 args, 可以通过 `vars(arg)` 转换成 dict 类型
    """
    def __init__(self, **kwargs):
        super(MyArgumentParser, self).__init__(**kwargs)
        self.add_argument('-c', '--config', type=str, required=True, help='config file path')

    def parse_args(self, args=None, namespace=None):
        args = super(MyArgumentParser, self).parse_args(args, namespace)
        new_conf = vars(args)

        with open(args.config, "r") as setting:
            config = yaml.load(setting, Loader=yaml.SafeLoader)
        for key, value in new_conf.items():
            config[key] = value

        return Namespace(**config)


def print_args(args):
    config = vars(args)
    print("**************** CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** CONFIGURATION ****************")
