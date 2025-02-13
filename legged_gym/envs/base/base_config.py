import inspect

"""
BaseConfig 的设计目的是递归地初始化它自己的所有成员类。
它会遍历自身的所有属性（包括子属性），
如果某个属性是类类型（通过inspect.isclass判断），它会将该属性替换为类的实例。
"""
class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        """
        遍历属性 (dir(obj))：

        使用 dir(obj) 获取对象的所有属性名。
        对于每个属性名，通过 getattr(obj, key) 获取对应的属性对象。
        跳过内置属性：

        使用条件 if key == "__class__": 跳过内置的 __class__ 属性。
        可以扩展这个规则以跳过更多内置属性，比如 key.startswith("__")（注释掉的代码部分）。
        判断是否是类 (inspect.isclass)：

        inspect.isclass(var) 检查当前属性是否是类定义。
        如果是类定义，则将其替换为一个实例。
        实例化类并替换属性：

        使用 var() 创建类的实例。
        使用 setattr(obj, key, i_var) 将原来的类替换为其实例。
        递归处理：

        对新创建的实例再次调用 init_member_classes，初始化它的成员类。
        """
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key=="__class__":
                continue
            # get the corresponding attribute object
            var =  getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)