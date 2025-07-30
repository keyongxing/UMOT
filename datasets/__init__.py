
from .dance import build as build_e2e_dance  # DanceTrack数据集端到端训练模式构建器
from .joint import build as build_e2e_joint  # 多数据集联合训练模式构建器


def build_dataset(image_set, args):
    """根据配置参数动态构建数据集

    核心功能：
    - 支持多模式数据集构建（训练/验证/测试）
    - 实现不同数据集类型的动态切换
    - 提供扩展接口支持新增数据集类型

    参数：
        image_set (str):
            - 数据集模式标识符，取值为 'train'/'val'/'test'
            - 用于区分不同阶段的数据集构建逻辑

        args (argparse.Namespace):
            - 包含完整配置参数的对象
            - 关键字段说明：
                * dataset_file: 数据集类型标识符（如'e2e_joint'/'e2e_dance'）
                * mot_path: MOT数据集存储路径
                * append_crowd: 是否包含CrowdHuman静态数据
                * det_db: 预生成检测结果数据库路径
                * sampler_steps: 课程学习阶段划分
                * sampler_lengths: 各阶段采样窗口长度

    返回：
        torch.utils.data2.Dataset:
            - 根据参数构建完成的数据集对象
            - 支持PyTorch数据加载器接口

    异常：
        ValueError:
            - 当传入的dataset_file参数不在支持列表时触发
            - 错误信息格式："dataset {dataset_file} not supported"

    设计亮点：
    - 模块化设计：通过子模块构建不同数据集类型
    - 扩展友好：通过条件分支实现插件式数据集加载
    - 配置驱动：完全依赖参数对象控制数据集构建流程
    """
    # 多数据集联合训练模式（例如DanceTrack + BDD100K）
    if args.dataset_file == 'e2e_joint':
        return build_e2e_joint(image_set, args)

    # DanceTrack数据集端到端跟踪模式
    if args.dataset_file == 'e2e_dance':
        return build_e2e_dance(image_set, args)

    # # 扩展建议：新增数据集需添加如下分支
    # if args.dataset_file == 'e2e_custom':
    #     from .custom import build as build_e2e_custom
    #     return build_e2e_custom(image_set, args)

    # 不支持的dataset_file参数处理
    raise ValueError(f'dataset {args.dataset_file} not supported')

# 扩展建议：新增数据集需添加如下分支
# if args.dataset_file == 'e2e_custom':
#     from .custom import build as build_e2e_custom
#     return build_e2e_custom(image_set, args)
