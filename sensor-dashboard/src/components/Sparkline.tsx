import type { KeyboardEvent } from 'react';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';

interface SparklineProps {
  label: string;
  history: { value: number }[];
  onClick?: () => void;
  isActive?: boolean;
}

const Sparkline = ({ label, history, onClick, isActive = false }: SparklineProps) => {

    const latestValue = history[history.length - 1]?.value || 0;
    const isInteractive = Boolean(onClick);
    const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
        if (!isInteractive || !onClick) {
            return;
        }
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onClick();
        }
    };

    return (
    <div
        style={{
            border: '1px solid #ddd',
            borderColor: isActive ? '#8884d8' : '#ddd',
            borderRadius: '8px',
            padding: '10px',
            textAlign: 'left',
            backgroundColor: 'white',
            cursor: isInteractive ? 'pointer' : 'default',
            boxShadow: isActive ? '0 0 0 2px rgba(136, 132, 216, 0.25)' : 'none',
            transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
        }}
        onClick={onClick}
        onKeyDown={handleKeyDown}
        role={isInteractive ? 'button' : undefined}
        tabIndex={isInteractive ? 0 : undefined}
        aria-expanded={isInteractive ? isActive : undefined}
        aria-label={isInteractive ? `Open ${label} chart` : undefined}
    >
        {/* 1. The Header Info */}
        <div style={{ fontSize: '12px', color: '#666' }}>{label}</div>
        <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
        {latestValue.toFixed(2)}
        </div>

        {/* 2. The Tiny Chart 📉 */}
        <div style={{ width: '100%', height: 50 }}>
        <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history}>
            <YAxis domain={['dataMin', 'dataMax']} hide={true} />
            <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#8884d8" 
                strokeWidth={2} 
                dot={false} 
            />

            </LineChart>
        </ResponsiveContainer>
        </div>
    </div>
    );
};

export default Sparkline;
