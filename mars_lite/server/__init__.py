"""
Server module for Trade RL

WebSocket/REST API for training監視・モデル管理・シグナル配信
(mars_lite.server.metrics_server、scripts/run_server.py から起動)。
mars_lite.server.signal_server は/api/signal/latestのみに絞った
軽量版（本番シグナル配信専用、現状は未配線・将来の縮小移行候補）。
"""
