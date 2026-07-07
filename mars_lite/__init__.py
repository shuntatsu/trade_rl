"""
Trade RL (mars_lite) - ポートフォリオ配分RLエージェント

複数銘柄の目標ウェイトを逐次決定するRLエージェント。多時間軸特徴・
オーダーフロー・funding rateを観測に使い、コスト控除後リターンを最大化する。
設計の正典は docs/ARCHITECTURE.md。
"""

__version__ = "0.1.0"
