import React, { useState } from 'react';
import { authenticatedFetch } from '../../services/api';

interface DatasetModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

export const DatasetModal: React.FC<DatasetModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'kaggle'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [kaggleInput, setKaggleInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  if (!isOpen) return null;

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsLoading(true);
    setMessage(null);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await authenticatedFetch('/api/dataset/upload', {
        method: 'POST',
        body: formData,
      });

      let data: any = {};
      try {
        data = await res.json();
      } catch {
        data = {};
      }

      if (!res.ok) {
        throw new Error(data.detail?.message || data.error || data.message || `Upload failed with status ${res.status}`);
      }

      setMessage({
        type: 'success',
        text: data.message || `Successfully imported dataset table: ${data.table_name || 'Uploaded Data'}`,
      });
      setFile(null);
      setTimeout(() => {
        onSuccess?.();
      }, 1500);
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to upload dataset' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKaggleImport = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!kaggleInput.trim()) return;

    setIsLoading(true);
    setMessage(null);
    try {
      const res = await authenticatedFetch('/api/dataset/kaggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url_or_code: kaggleInput.trim() }),
      });

      let data: any = {};
      try {
        data = await res.json();
      } catch {
        data = {};
      }

      if (!res.ok) {
        throw new Error(data.detail?.message || data.error || data.message || `Kaggle import failed with status ${res.status}`);
      }

      setMessage({
        type: 'success',
        text: data.message || `Imported dataset successfully from Kaggle!`,
      });
      setKaggleInput('');
      setTimeout(() => {
        onSuccess?.();
      }, 1500);
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message || 'Failed to import Kaggle dataset' });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-surface-container-low border border-outline-variant rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-outline-variant/60 flex items-center justify-between bg-surface-container/50">
          <div className="flex items-center gap-2 text-primary">
            <span className="material-symbols-outlined">dataset</span>
            <h3 className="font-headline-sm text-base font-bold text-on-surface">Add Dataset</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high transition-colors cursor-pointer"
          >
            <span className="material-symbols-outlined text-sm">close</span>
          </button>
        </div>

        {/* Tab Selection */}
        <div className="flex border-b border-outline-variant/60 bg-surface-container-low">
          <button
            onClick={() => { setActiveTab('upload'); setMessage(null); }}
            className={`flex-1 py-3 text-xs font-semibold uppercase tracking-wider flex items-center justify-center gap-2 border-b-2 transition-colors cursor-pointer ${
              activeTab === 'upload'
                ? 'border-primary text-primary bg-primary/5'
                : 'border-transparent text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <span className="material-symbols-outlined text-sm">upload_file</span>
            <span>Upload File (ZIP / CSV)</span>
          </button>
          <button
            onClick={() => { setActiveTab('kaggle'); setMessage(null); }}
            className={`flex-1 py-3 text-xs font-semibold uppercase tracking-wider flex items-center justify-center gap-2 border-b-2 transition-colors cursor-pointer ${
              activeTab === 'kaggle'
                ? 'border-primary text-primary bg-primary/5'
                : 'border-transparent text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <span className="material-symbols-outlined text-sm">cloud_download</span>
            <span>Kaggle Code / URL</span>
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-4">
          {message && (
            <div
              className={`p-3 rounded-xl text-xs font-medium flex items-center gap-2 ${
                message.type === 'success'
                  ? 'bg-secondary/15 border border-secondary/40 text-secondary'
                  : 'bg-error-container/30 border border-error-container/60 text-error'
              }`}
            >
              <span className="material-symbols-outlined text-sm">
                {message.type === 'success' ? 'check_circle' : 'error'}
              </span>
              <span>{message.text}</span>
            </div>
          )}

          {activeTab === 'upload' ? (
            <form onSubmit={handleFileUpload} className="space-y-4">
              <div className="border-2 border-dashed border-outline-variant/80 rounded-2xl p-6 flex flex-col items-center justify-center bg-surface-container-lowest hover:border-primary/50 transition-colors text-center cursor-pointer">
                <span className="material-symbols-outlined text-3xl text-primary mb-2">folder_zip</span>
                <p className="text-xs font-semibold text-on-surface">
                  {file ? file.name : 'Select or drag & drop a dataset file'}
                </p>
                <p className="text-[11px] text-on-surface-variant mt-1">
                  Supports ZIP archives containing CSVs, direct CSV files, or SQLite databases.
                </p>
                <input
                  type="file"
                  accept=".zip,.csv,.sqlite,.db"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  className="mt-3 text-xs text-on-surface-variant file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-primary/20 file:text-primary hover:file:bg-primary/30 cursor-pointer"
                />
              </div>

              <div className="flex gap-3 justify-end pt-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2 rounded-xl bg-surface-container-high text-on-surface text-xs font-semibold hover:brightness-110 transition-all cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!file || isLoading}
                  className="px-5 py-2 rounded-xl bg-primary text-on-primary text-xs font-bold uppercase tracking-wider hover:brightness-110 transition-all disabled:opacity-40 cursor-pointer shadow-sm"
                >
                  {isLoading ? 'Ingesting...' : 'Import Dataset'}
                </button>
              </div>
            </form>
          ) : (
            <form onSubmit={handleKaggleImport} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-on-surface-variant mb-1.5">
                  Kaggle Dataset Code or URL
                </label>
                <textarea
                  value={kaggleInput}
                  onChange={(e) => setKaggleInput(e.target.value)}
                  placeholder='e.g. "https://www.kaggle.com/datasets/user/dataset-name" or "kaggle datasets download -d user/dataset-name"'
                  rows={3}
                  className="w-full bg-surface-container-lowest border border-outline-variant/80 rounded-xl px-3 py-2 text-xs text-on-surface focus:outline-none focus:border-primary font-mono custom-scrollbar"
                />
              </div>

              <div className="flex gap-3 justify-end pt-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2 rounded-xl bg-surface-container-high text-on-surface text-xs font-semibold hover:brightness-110 transition-all cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!kaggleInput.trim() || isLoading}
                  className="px-5 py-2 rounded-xl bg-primary text-on-primary text-xs font-bold uppercase tracking-wider hover:brightness-110 transition-all disabled:opacity-40 cursor-pointer shadow-sm"
                >
                  {isLoading ? 'Downloading...' : 'Fetch Dataset'}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};
