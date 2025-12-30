import { LineChart, Line, ResponsiveContainer , YAxis} from 'recharts';

interface SparklineProps {
  label: string;
  history: { value: number }[];
}

const Sparkline = ({ label, history }: SparklineProps) => {

    const latestValue = history[history.length - 1]?.value || 0;

    return (
    <div style={{ 
        border: '1px solid #ddd', 
        borderRadius: '8px', 
        padding: '10px', 
        textAlign: 'left',
        backgroundColor: 'white' 
    }}>
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