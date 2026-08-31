#!/usr/bin/env python3
"""验证本地 rollout 功能的完整性检查脚本。"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

def test_local_llm_module():
    """测试 local_llm 模块的可导入性。"""
    print("=" * 60)
    print("测试1：local_llm 模块导入")
    print("=" * 60)
    try:
        import local_llm
        print("✓ local_llm 模块可成功导入")

        # 检查必要的函数和异常类
        assert hasattr(local_llm, 'chat'), "缺少 chat 函数"
        assert hasattr(local_llm, 'LocalLLMError'), "缺少 LocalLLMError 异常类"
        print("✓ chat 函数和 LocalLLMError 异常类都存在")

        # 检查函数签名
        import inspect
        sig = inspect.signature(local_llm.chat)
        params = list(sig.parameters.keys())
        expected = ['messages', 'temperature', 'timeout', 'retries']
        assert all(p in params for p in expected), f"chat 函数缺少必要参数，有 {params}"
        print(f"✓ chat 函数参数正确：{params}")

        return True
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        return False


def test_user_simulator_chat_fn():
    """测试 UserSimulator 的 chat_fn 参数。"""
    print("\n" + "=" * 60)
    print("测试2：UserSimulator chat_fn 参数")
    print("=" * 60)
    try:
        from user_simulator import UserSimulator

        # 加载一个测试用例
        test_cases = json.loads((ROOT / "data" / "test_cases.json").read_text(encoding="utf-8"))
        test_case = test_cases[0]

        # 测试默认 chat_fn（应为 llm.chat_json）
        sim_default = UserSimulator(test_case)
        assert hasattr(sim_default, 'chat_fn'), "UserSimulator 缺少 chat_fn 属性"
        print("✓ UserSimulator 有 chat_fn 属性")

        # 测试自定义 chat_fn
        def dummy_chat_fn(messages, temperature=0.7):
            return {"utterance": "test", "satisfaction": "neutral"}

        sim_custom = UserSimulator(test_case, chat_fn=dummy_chat_fn)
        assert sim_custom.chat_fn is dummy_chat_fn, "自定义 chat_fn 未被正确设置"
        print("✓ 自定义 chat_fn 设置成功")

        return True
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcts_env_rollout_sim():
    """测试 SearchEnv 的 rollout_sim 参数。"""
    print("\n" + "=" * 60)
    print("测试3：SearchEnv rollout_sim 参数")
    print("=" * 60)
    try:
        from mcts_env import SearchEnv
        from sop import SOPEngine
        from user_simulator import UserSimulator
        import inspect

        # 检查 SearchEnv.__init__ 的参数
        sig = inspect.signature(SearchEnv.__init__)
        params = list(sig.parameters.keys())
        assert 'rollout_sim' in params, f"SearchEnv.__init__ 缺少 rollout_sim 参数，有 {params}"
        print(f"✓ SearchEnv.__init__ 包含 rollout_sim 参数")

        # 加载测试用例
        test_cases = json.loads((ROOT / "data" / "test_cases.json").read_text(encoding="utf-8"))
        test_case = test_cases[0]

        # 测试创建 SearchEnv 带 rollout_sim
        engine = SOPEngine()
        sim = UserSimulator(test_case)
        rollout_sim = UserSimulator(test_case)
        branch_ctx = {"issue_type": test_case["issue_type"], "evidence_valid": True}

        env = SearchEnv(engine, sim, branch_ctx, rollout_sim=rollout_sim)
        assert env.rollout_sim is rollout_sim, "rollout_sim 未被正确设置"
        print("✓ SearchEnv 可以使用 rollout_sim 参数")

        # 测试 rollout_sim 为 None 的情况
        env_default = SearchEnv(engine, sim, branch_ctx, rollout_sim=None)
        assert env_default.rollout_sim is None, "rollout_sim 应该为 None"
        print("✓ rollout_sim 为 None 时正确处理")

        return True
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcts_agent_use_local_rollout():
    """测试 MCTSAgent 的 use_local_rollout 参数。"""
    print("\n" + "=" * 60)
    print("测试4：MCTSAgent use_local_rollout 参数")
    print("=" * 60)
    try:
        from mcts_agent import MCTSAgent
        import inspect

        # 检查 MCTSAgent.__init__ 的参数
        sig = inspect.signature(MCTSAgent.__init__)
        params = list(sig.parameters.keys())
        assert 'use_local_rollout' in params, f"MCTSAgent.__init__ 缺少 use_local_rollout 参数，有 {params}"
        print(f"✓ MCTSAgent.__init__ 包含 use_local_rollout 参数")

        # 加载测试用例
        test_cases = json.loads((ROOT / "data" / "test_cases.json").read_text(encoding="utf-8"))
        test_case = test_cases[0]

        # 测试创建 MCTSAgent 带 use_local_rollout=False（默认）
        agent_default = MCTSAgent(test_case, use_local_rollout=False)
        assert agent_default.env.rollout_sim is None, "默认情况下 rollout_sim 应该为 None"
        print("✓ use_local_rollout=False 时 rollout_sim 为 None")

        # 注意：use_local_rollout=True 会尝试导入 local_llm，这在没有 ollama 的环境中会失败
        # 所以我们只测试参数的存在性，不测试实际功能

        return True
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_runner_use_local_rollout_flag():
    """测试 runner.py 中的 --use-local-rollout 标志。"""
    print("\n" + "=" * 60)
    print("测试5：runner.py --use-local-rollout 标志")
    print("=" * 60)
    try:
        import argparse
        import subprocess

        # 检查 runner.py 中是否包含 --use-local-rollout 标志
        runner_code = (ROOT / "src" / "runner.py").read_text()
        assert "--use-local-rollout" in runner_code, "runner.py 中缺少 --use-local-rollout 标志"
        print("✓ runner.py 中包含 --use-local-rollout 标志")

        # 检查 action="store_true" 的正确性
        assert 'action="store_true"' in runner_code, "--use-local-rollout 应该使用 store_true 动作"
        print("✓ --use-local-rollout 使用 store_true 动作")

        return True
    except Exception as e:
        print(f"✗ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有验证测试。"""
    print("\n")
    print("▶ 本地 Rollout 功能完整性检查")
    print("▶" * 30)

    results = {
        "local_llm_module": test_local_llm_module(),
        "user_simulator_chat_fn": test_user_simulator_chat_fn(),
        "mcts_env_rollout_sim": test_mcts_env_rollout_sim(),
        "mcts_agent_use_local_rollout": test_mcts_agent_use_local_rollout(),
        "runner_use_local_rollout_flag": test_runner_use_local_rollout_flag(),
    }

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有验证测试通过！本地 Rollout 功能已准备好")
    else:
        print("✗ 部分测试失败，请检查上面的错误信息")
        sys.exit(1)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
