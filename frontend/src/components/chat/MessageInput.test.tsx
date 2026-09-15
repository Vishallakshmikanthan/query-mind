import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MessageInput } from './MessageInput';

describe('MessageInput', () => {
  it('submits typed message', () => {
    const onSend = vi.fn();
    const onStop = vi.fn();
    render(<MessageInput onSend={onSend} onStop={onStop} disabled={false} status="idle" />);

    const textarea = screen.getByPlaceholderText(/Ask anything about your data/i);
    fireEvent.change(textarea, { target: { value: 'Show orders' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(onSend).toHaveBeenCalledWith('Show orders');
  });

  it('does not submit on Shift+Enter', () => {
    const onSend = vi.fn();
    render(<MessageInput onSend={onSend} onStop={vi.fn()} disabled={false} status="idle" />);

    const textarea = screen.getByPlaceholderText(/Ask anything about your data/i);
    fireEvent.change(textarea, { target: { value: 'line 1' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

    expect(onSend).not.toHaveBeenCalled();
  });

  it('shows stop button while streaming', () => {
    const onStop = vi.fn();
    render(<MessageInput onSend={vi.fn()} onStop={onStop} disabled status="streaming" />);

    const stopButton = screen.getByRole('button', { name: /stop/i });
    expect(stopButton).toBeInTheDocument();
    fireEvent.click(stopButton);
    expect(onStop).toHaveBeenCalled();
  });

  it('sends quick prompt when chip is clicked', () => {
    const onSend = vi.fn();
    render(<MessageInput onSend={onSend} onStop={vi.fn()} disabled={false} status="idle" />);

    const chip = screen.getByText(/top 5 products by revenue/i);
    fireEvent.click(chip);

    expect(onSend).toHaveBeenCalledWith('Top 5 products by revenue as a chart');
  });
});
