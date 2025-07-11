import requests
import numpy as np
from typing import Dict, Tuple, Any
import json
from VLABench.evaluation.model.policy.base import Policy
class RemoteAgentClient(Policy):
        """
        初始化远程代理客户端
        
        Args:
            server_url: 服务器URL地址
        """
        def __init__(self, 
                **kwargs):
            super().__init__(**kwargs)
