import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES
import copy


@PIPELINES.register_module()
class Compose(object):

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        
        if isinstance(transforms, list):
            self.transforms_type = 'list'
        elif isinstance(transforms, tuple): 
            self.transforms_type = 'tuple'
        else:
            NotImplementedError

        if self.transforms_type=='list':
            self.transforms = []
            for transform in transforms:
                if isinstance(transform, dict):
                    transform = build_from_cfg(transform, PIPELINES)
                    self.transforms.append(transform)
                elif callable(transform):
                    self.transforms.append(transform)
                else:
                    raise TypeError('transform must be callable or a dict')
        elif self.transforms_type=='tuple':
            self.transforms = []
            for transforms_i in transforms:
                transforms_i_builded = []
                for transform in transforms_i:
                    if isinstance(transform, dict):
                        transform = build_from_cfg(transform, PIPELINES)
                        transforms_i_builded.append(transform)
                    elif callable(transform):
                        transforms_i_builded.append(transform)
                    else:
                        raise TypeError('transform must be callable or a dict')
                self.transforms.append(transforms_i_builded)

    def __call__(self, data):
        if self.transforms_type=='list':
            for t in self.transforms:
                data = t(data)
                if data is None:
                    return None
            return data
        elif self.transforms_type=='tuple':
            # share pipline
            for t in self.transforms[0]:
                data = t(data)
                if data is None:
                    return None
            # train pipline and aug train pipline
            data_return = []
            for transforms_i in self.transforms[1:]:
                data_copy = copy.deepcopy(data)
                for t in transforms_i:
                    data_copy = t(data_copy)
                    if data_copy is None:
                        return None
                data_return.append(data_copy)
            return data_return 

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
