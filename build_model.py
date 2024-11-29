import json
from models.vmamba import VSSM

def build_vssm_model(config, is_pretrain=False):
    model = VSSM(
        patch_size=4,
        in_chans=3,
        num_classes=2,
        depths=[ 2, 2, 15, 2 ],
        dims=128,
        # ===================
        ssm_d_state=1,
        ssm_ratio=2.0,
        # ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        # ===================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # ===================
        drop_path_rate=0.6,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        # gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=False,
        # ===================
        posembed=False,
        imgsize=224,
    )
    return model


def build_model(is_pretrain=False):
    model = build_vssm_model(is_pretrain)
    return model