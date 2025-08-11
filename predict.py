import functools
from os import path as osp
import h5py
import numpy as np
import paddle
import ppsci
import hydra
from omegaconf import DictConfig
import utils as fourcast_utils
from ppsci.utils import logger
from typing import Tuple


def get_vis_data(
        file_path: str,
        date_strings: Tuple[str, ...],
        num_timestamps: int,
        vars_channel: Tuple[int, ...],
        img_h: int,
        data_mean: np.ndarray,
        data_std: np.ndarray,
        output_dir: str,
):
    """读取输入数据，并进行标准化处理"""
    _file = h5py.File(file_path, "r")["fields"]
    data = []
    for date_str in date_strings:
        hours_since_jan_01_epoch = fourcast_utils.date_to_hours(date_str)
        ic = int(hours_since_jan_01_epoch / 1)
        data.append(_file[ic: ic + num_timestamps + 1, vars_channel, 0:img_h])
    data = np.asarray(data)

    vis_data = {"input": (data[:, 0] - data_mean) / data_std}
    return vis_data


def evaluate(cfg: DictConfig):
    """用于预测的评估函数，不与真实值进行比较"""

    # 设置随机种子确保结果可复现
    ppsci.utils.misc.set_random_seed(cfg.seed)

    # 初始化日志记录器
    logger.init_logger("ppsci", osp.join(cfg.output_dir, "eval.log"), "info")

    # 加载均值和标准差
    data_mean, data_std = fourcast_utils.get_mean_std(
        cfg.DATA_MEAN_PATH, cfg.DATA_STD_PATH, cfg.VARS_CHANNEL
    )

    # 设置模型配置
    model_cfg = dict(cfg.MODEL.afno)
    model_cfg.update(
        {"output_keys": tuple(f"output_{i}" for i in range(cfg.EVAL.num_timestamps)),
         "num_timestamps": cfg.EVAL.num_timestamps}
    )

    # 初始化模型
    model = ppsci.arch.AFNONet(**model_cfg)

    # 设置数据处理的变换流程
    transforms = [
        {"SqueezeData": {}},
        {"Normalize": {"mean": data_mean, "std": data_std}},
    ]

    # 设置用于预测的数据集配置
    eval_dataloader_cfg = {
        "dataset": {
            "name": "ERA5Dataset",
            "file_path": cfg.TEST_FILE_PATH,
            "input_keys": cfg.MODEL.afno.input_keys,
            "label_keys": [],  # 无需真实值标签
            "vars_channel": cfg.VARS_CHANNEL,
            "transforms": transforms,
            "num_label_timestamps": cfg.EVAL.num_timestamps,
            "training": False,
            "stride": 8,
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "batch_size": cfg.EVAL.batch_size,
    }

    # 设置用于保存预测结果的函数
    predictions = []

    def output_wind_func(d, var_name, data_mean, data_std):
        output = (d[var_name] * data_std) + data_mean  # 将标准化后的数据还原
        wind_data = []
        for i in range(output.shape[0]):
            wind_data.append(output[i][0])
        predictions.append(wind_data)
        np.save(osp.join(cfg.output_dir, "predictions.npy"), np.array(predictions))  # 保存预测结果
        return paddle.to_tensor(wind_data, paddle.get_default_dtype())

    # 可视化表达式，用于生成和保存结果
    vis_output_expr = {}
    for i in range(cfg.EVAL.num_timestamps):
        hour = (i + 1) * 1
        vis_output_expr[f"output_{hour}h"] = functools.partial(
            output_wind_func,
            var_name=f"output_{i}",
            data_mean=paddle.to_tensor(data_mean, paddle.get_default_dtype()),
            data_std=paddle.to_tensor(data_std, paddle.get_default_dtype()),
        )

    # 获取输入数据
    DATE_STRINGS = ("2024-01-01 00:00:00",)  # 可以替换成实际的时间戳
    vis_data = get_vis_data(
        cfg.TEST_FILE_PATH,
        DATE_STRINGS,
        cfg.EVAL.num_timestamps,
        cfg.VARS_CHANNEL,
        cfg.IMG_H,
        data_mean,
        data_std,
        cfg.output_dir
    )

    # 初始化模型求解器，不再需要验证器
    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        visualizer=None,  # 实际应用中无需可视化
        pretrained_model_path=cfg.EVAL.pretrained_model_path,
    )

    # 执行预测
    solver.eval()

    # 保存预测结果
    solver.visualize(vis_output_expr, vis_data)  # 直接可视化预测结果


@hydra.main(
    version_base=None, config_path="./conf", config_name="fourcastnet_finetune.yaml"
)
def main(cfg: DictConfig):
    if cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode 应为 'eval'，但得到 '{cfg.mode}'")


if __name__ == "__main__":
    main()
