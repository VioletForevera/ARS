#!/usr/bin/env python3
"""
根目录入口脚本

目的：
- 允许在项目根目录直接运行：
    python run_mountaincar.py ...
- 内部转发到包内的真正实现：mountaincar_cl.run_mountaincar.main
"""

from mountaincar_cl.run_mountaincar import main


if __name__ == "__main__":
    main()


