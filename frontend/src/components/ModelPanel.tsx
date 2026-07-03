import React, { useState, useEffect } from 'react';
import { FolderOpen, Trash2, RefreshCw, Clock, HardDrive } from 'lucide-react';

interface ModelInfo {
    id: string;
    path: string;
    size_bytes: number;
    modified_at: number;
    metadata?: {
        total_timesteps?: number;
        mean_reward?: number;
        std_reward?: number;
        learning_rate?: number;
        notes?: string;
        tags?: string[];
    };
}

interface ModelPanelProps {
    apiUrl?: string;
    onModelSelect?: (modelId: string) => void;
}

/**
 * モデル管理パネル
 * 
 * 保存済みモデルの一覧表示、読み込み、削除を提供。
 */
export const ModelPanel: React.FC<ModelPanelProps> = ({
    apiUrl = 'http://localhost:8001',
    onModelSelect,
}) => {
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedModel, setSelectedModel] = useState<string | null>(null);

    // モデル一覧を取得
    const fetchModels = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(`${apiUrl}/api/models`);
            if (!response.ok) {
                throw new Error('Failed to fetch models');
            }
            const data = await response.json();
            setModels(data.models || []);
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    };

    // 初回読み込み
    useEffect(() => {
        fetchModels();
    }, [apiUrl]);

    // モデル削除
    const handleDelete = async (modelId: string) => {
        if (!confirm(`モデル "${modelId}" を削除しますか？`)) {
            return;
        }

        try {
            const response = await fetch(`${apiUrl}/api/models/${modelId}`, {
                method: 'DELETE',
            });
            if (!response.ok) {
                throw new Error('Failed to delete model');
            }
            await fetchModels();
        } catch (e) {
            setError(e instanceof Error ? e.message : 'Delete failed');
        }
    };

    // モデル選択
    const handleSelect = (modelId: string) => {
        setSelectedModel(modelId);
        onModelSelect?.(modelId);
    };

    // ファイルサイズをフォーマット
    const formatSize = (bytes: number): string => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    // 日付をフォーマット
    const formatDate = (timestamp: number): string => {
        return new Date(timestamp * 1000).toLocaleString('ja-JP', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    return (
        <div className="card h-full flex flex-col">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <FolderOpen size={16} className="text-primary" />
                    <h3 className="text-sm font-semibold">SAVED MODELS</h3>
                </div>
                <button
                    onClick={fetchModels}
                    disabled={loading}
                    className="p-1.5 rounded hover:bg-white/5 transition-colors"
                    title="Refresh"
                >
                    <RefreshCw size={14} className={`text-muted ${loading ? 'animate-spin' : ''}`} />
                </button>
            </div>

            {error && (
                <div className="text-red-400 text-xs mb-2 p-2 bg-red-500/10 rounded">
                    {error}
                </div>
            )}

            <div className="flex-1 overflow-y-auto space-y-2">
                {models.length === 0 ? (
                    <div className="text-muted text-sm text-center py-8">
                        {loading ? 'Loading...' : 'No models found'}
                    </div>
                ) : (
                    models.map((model) => (
                        <div
                            key={model.id}
                            onClick={() => handleSelect(model.id)}
                            className={`p-3 rounded border transition-all cursor-pointer ${selectedModel === model.id
                                ? 'border-primary bg-primary/10'
                                : 'border-border hover:border-primary/50 bg-surface/50'
                                }`}
                        >
                            <div className="flex items-start justify-between">
                                <div className="flex-1 min-w-0">
                                    <div className="font-mono text-sm truncate" title={model.id}>
                                        {model.id}
                                    </div>
                                    <div className="flex items-center gap-3 mt-1 text-xs text-muted">
                                        <span className="flex items-center gap-1">
                                            <HardDrive size={10} />
                                            {formatSize(model.size_bytes)}
                                        </span>
                                        <span className="flex items-center gap-1">
                                            <Clock size={10} />
                                            {formatDate(model.modified_at)}
                                        </span>
                                    </div>

                                    {model.metadata && (
                                        <div className="mt-2 text-xs">
                                            {model.metadata.mean_reward !== undefined && (
                                                <span className="text-primary mr-3">
                                                    Reward: {model.metadata.mean_reward.toFixed(2)}
                                                </span>
                                            )}
                                            {model.metadata.total_timesteps !== undefined && (
                                                <span className="text-muted">
                                                    {(model.metadata.total_timesteps / 1000).toFixed(0)}k steps
                                                </span>
                                            )}
                                        </div>
                                    )}
                                </div>

                                <div className="flex items-center gap-1 ml-2">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleDelete(model.id);
                                        }}
                                        className="p-1.5 rounded hover:bg-red-500/20 transition-colors"
                                        title="Delete"
                                    >
                                        <Trash2 size={14} className="text-red-400" />
                                    </button>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {selectedModel && (
                <div className="mt-4 pt-4 border-t border-border">
                    <div className="text-xs text-muted mb-2">Selected:</div>
                    <div className="font-mono text-sm text-primary truncate">
                        {selectedModel}
                    </div>
                </div>
            )}
        </div>
    );
};
