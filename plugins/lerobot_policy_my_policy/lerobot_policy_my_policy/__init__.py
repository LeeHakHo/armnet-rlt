"""Example LeRobot policy plugin; importing it registers the "my_policy" type."""

from .configuration_my_policy import MyPolicyConfig
from .modeling_my_policy import MyPolicyPolicy
from .processor_my_policy import make_my_policy_pre_post_processors

__all__ = ["MyPolicyConfig", "MyPolicyPolicy", "make_my_policy_pre_post_processors"]
