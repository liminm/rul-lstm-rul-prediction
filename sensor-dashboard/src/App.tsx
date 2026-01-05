import { useEffect, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './App.css';
import Sparkline from './components/Sparkline';

const SENSOR_NAMES = {
  s_1: 'Fan Inlet Temp',
  s_2: 'LPC Outlet Temp',
  s_3: 'HPC Outlet Temp',
  s_4: 'LPT Outlet Temp',
  s_5: 'Fan Inlet Pressure',
  s_6: 'Bypass Duct Press',
  s_7: 'HPC Outlet Press',
  s_8: 'Phys Fan Speed',
  s_9: 'Phys Core Speed',
  s_10: 'Engine Press Ratio',
  s_11: 'Static HPC Outlet P',
  s_12: 'Fuel Flow Ratio',
  s_13: 'Corr Fan Speed',
  s_14: 'Corr Core Speed',
  s_15: 'Bypass Ratio',
  s_16: 'Burner Burner Ratio',
  s_17: 'Bleed Enthalpy',
  s_18: 'Demanded Fan Speed',
  s_19: 'Demanded Corr Fan Speed',
  s_20: 'HPT Coolant Bleed',
  s_21: 'LPT Coolant Bleed',
};


const SETTING_NAMES = {
  setting_1: 'Altitude',
  setting_2: 'Mach Number',
  setting_3: 'Throttle Resolver Angle',
};

const ALL_SENSOR_NAMES = { ...SENSOR_NAMES, ...SETTING_NAMES };

type SparkPoint = {
  value: number;
};

type SensorCard = {
  id: number;
  sensorKey: string;
  label: string;
  history: SparkPoint[];
};

type SensorStats = {
  min: number | null;
  max: number | null;
  avg: number | null;
  latest: number | null;
};

type PredictionContext = {
  rulPercentile: number | null;
  healthBand: string | null;
  healthBandLevel: string | null;
  observedCycles: number | null;
  trainRulMin: number | null;
  trainRulMax: number | null;
  trainRulMedian: number | null;
  trainCycleMedian: number | null;
  trainCycleMean: number | null;
};

const computeStats = (history: SparkPoint[]): SensorStats => {
  if (history.length === 0) {
    return { min: null, max: null, avg: null, latest: null };
  }

  let min = history[0].value;
  let max = history[0].value;
  let sum = 0;

  for (const point of history) {
    min = Math.min(min, point.value);
    max = Math.max(max, point.value);
    sum += point.value;
  }

  return {
    min,
    max,
    avg: sum / history.length,
    latest: history[history.length - 1].value,
  };
};

const formatMetric = (value: number | null) => {
  if (value === null || Number.isNaN(value)) {
    return 'N/A';
  }
  return value.toFixed(2);
};

function App() {
  const [selectedEngine, setSelectedEngine] = useState(1);
  const [availableEngines, setAvailableEngines] = useState<number[]>([]);
  const [predictedRul, setPredictedRul] = useState<number | null>(null);
  const [trueRul, setTrueRul] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [lastRun, setLastRun] = useState<string | null>(null);
  const [predictionContext, setPredictionContext] =
    useState<PredictionContext | null>(null);
  const [sensors, setSensors] = useState<SensorCard[]>([]);
  const [activeSensorKey, setActiveSensorKey] = useState<string | null>(null);

  const activeSensor =
    sensors.find((sensor) => sensor.sensorKey === activeSensorKey) ?? null;

  useEffect(() => {
    setActiveSensorKey(null);
    setPredictedRul(null);
    setTrueRul(null);
    setPredictionError(null);
    setLastRun(null);
    setPredictionContext(null);
  }, [selectedEngine]);

  useEffect(() => {
    const fetchEngines = async () => {
      try {
        const response = await fetch('/engines/');
        const data = await response.json();
        setAvailableEngines(data);
        if (data.length > 0) {
          setSelectedEngine(data[0]);
        }
      } catch (error) {
        console.error('Error fetching engines:', error);
      }
    };

    fetchEngines();
  }, []);

  useEffect(() => {
    if (!activeSensor) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setActiveSensorKey(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [activeSensor]);

  const handlePredict = async () => {
    setIsPredicting(true);
    setPredictionError(null);
    try {
      const response = await fetch('/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          unit_nr: selectedEngine,
        }),
      });
      if (!response.ok) {
        throw new Error(`Prediction failed with status ${response.status}`);
      }
      const data = await response.json();
      const predicted = Number(data.predicted_rul);
      const truth = data.true_rul === null ? null : Number(data.true_rul);
      setPredictedRul(Number.isNaN(predicted) ? null : predicted);
      setTrueRul(truth !== null && Number.isNaN(truth) ? null : truth);
      if (data.context) {
        setPredictionContext({
          rulPercentile:
            data.context.rul_percentile === null
              ? null
              : Number(data.context.rul_percentile),
          healthBand: data.context.health_band ?? null,
          healthBandLevel: data.context.health_band_level ?? null,
          observedCycles:
            data.context.observed_cycles === null
              ? null
              : Number(data.context.observed_cycles),
          trainRulMin:
            data.context.train_rul_min === null
              ? null
              : Number(data.context.train_rul_min),
          trainRulMax:
            data.context.train_rul_max === null
              ? null
              : Number(data.context.train_rul_max),
          trainRulMedian:
            data.context.train_rul_median === null
              ? null
              : Number(data.context.train_rul_median),
          trainCycleMedian:
            data.context.train_cycle_median === null
              ? null
              : Number(data.context.train_cycle_median),
          trainCycleMean:
            data.context.train_cycle_mean === null
              ? null
              : Number(data.context.train_cycle_mean),
        });
      } else {
        setPredictionContext(null);
      }
      setLastRun(new Date().toLocaleTimeString());
    } catch (error) {
      console.error('Error fetching RUL prediction:', error);
      setPredictionError('Prediction failed. Try again.');
    } finally {
      setIsPredicting(false);
    }
  };

  const fetchData = async () => {
    try {
      const response = await fetch(
        `/sensors/?limit=500&unit_nr=${selectedEngine}`,
      );

        
      const data = await response.json();

      const formattedSensors = Object.keys(ALL_SENSOR_NAMES).map((key, index) => ({
        id: index,
        sensorKey: key,
        label: ALL_SENSOR_NAMES[key as keyof typeof ALL_SENSOR_NAMES],
        history: data.map((row: Record<string, number>) => ({ value: row[key] })),
      }));

      setSensors(formattedSensors);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  useEffect(() => {
    fetchData();
  }, [selectedEngine]);

  const closeModal = () => setActiveSensorKey(null);
  const expandedHistory = activeSensor
    ? activeSensor.history.map((point, index) => ({
        cycle: index + 1,
        value: point.value,
      }))
    : [];
  const stats = activeSensor ? computeStats(activeSensor.history) : null;
  const delta =
    predictedRul !== null && trueRul !== null ? predictedRul - trueRul : null;
  const formatRulText = (value: number | null, fallback: string) => {
    if (value === null || Number.isNaN(value)) {
      return fallback;
    }
    return `${value.toFixed(2)} cycles`;
  };
  const deltaLabel =
    delta === null ? null : `${delta >= 0 ? '+' : ''}${delta.toFixed(2)} cycles`;
  const deltaTone = delta !== null && delta >= 0 ? 'positive' : 'negative';
  const percentileLabel =
    predictionContext?.rulPercentile === null || predictionContext?.rulPercentile === undefined
      ? 'N/A'
      : `P${Math.round(predictionContext.rulPercentile)}`;
  const healthLabel = predictionContext?.healthBand ?? 'N/A';
  const healthLevel = predictionContext?.healthBandLevel ?? 'neutral';
  const observedLabel =
    predictionContext?.observedCycles !== null && predictionContext?.observedCycles !== undefined
      ? `${Math.round(predictionContext.observedCycles)} cycles observed`
      : 'Observed cycles: N/A';
  const trainRangeLabel =
    predictionContext?.trainRulMin !== null &&
    predictionContext?.trainRulMax !== null &&
    predictionContext?.trainRulMin !== undefined &&
    predictionContext?.trainRulMax !== undefined
      ? `Train RUL range: ${Math.round(
          predictionContext.trainRulMin,
        )}-${Math.round(predictionContext.trainRulMax)} cycles`
      : null;
  const trainLifeLabel =
    predictionContext?.trainCycleMedian !== null &&
    predictionContext?.trainCycleMedian !== undefined
      ? `Median train life: ${Math.round(predictionContext.trainCycleMedian)} cycles`
      : null;
  const statusLabel = isPredicting
    ? 'Running...'
    : lastRun
      ? `Last run: ${lastRun}`
      : 'Not run yet';

  return (
    <>
      <h1>Engine Dashboard</h1>
      <div className="top-bar">
        <div className="engine-control">
          <label>Select Engine: </label>
          <select
            value={selectedEngine}
            onChange={(e) => setSelectedEngine(Number(e.target.value))}
            style={{ marginLeft: '10px', padding: '5px' }}
          >
            {availableEngines.map((engineNr) => (
              <option key={engineNr} value={engineNr}>
                Engine {engineNr}
              </option>
            ))}
          </select>
        </div>
        <div className="prediction-panel">
          <div className="prediction-actions">
            <button
              className="prediction-button"
              type="button"
              onClick={handlePredict}
              disabled={isPredicting}
            >
              {isPredicting ? 'Running...' : 'Run prediction'}
            </button>
            <div className="prediction-status">{statusLabel}</div>
            {predictionError && (
              <div className="prediction-error">{predictionError}</div>
            )}
          </div>
          <div className="prediction-cards">
            <div className="prediction-card">
              <div className="prediction-label">Predicted RUL</div>
              <div className="prediction-value">
                {formatRulText(predictedRul, 'Not run')}
              </div>
            </div>
            <div className="prediction-card">
              <div className="prediction-label">True RUL</div>
              <div className="prediction-value">
                {formatRulText(trueRul, 'N/A')}
              </div>
            </div>
          </div>
          {deltaLabel && (
            <div className={`prediction-delta ${deltaTone}`}>
              Delta {deltaLabel}
            </div>
          )}
          {predictionContext && (
            <>
              <div className="prediction-context">
                <div className={`prediction-chip prediction-band ${healthLevel}`}>
                  Health: {healthLabel}
                </div>
                <div className="prediction-chip">
                  RUL percentile: {percentileLabel}
                </div>
                <div className="prediction-chip">{observedLabel}</div>
              </div>
              <div className="prediction-footnote">
                {trainRangeLabel}
                {trainRangeLabel && trainLifeLabel ? ' | ' : ''}
                {trainLifeLabel}
              </div>
            </>
          )}
          {deltaLabel && (
            <div className="prediction-note">
              Positive delta means the model overestimated remaining life.
            </div>
          )}
        </div>
      </div>
      {activeSensor && (
        <div className="detail-panel">
          <div className="detail-panel-header">
            <div>
              <div className="detail-panel-title">{activeSensor.label}</div>
              <div className="detail-panel-subtitle">
                {activeSensor.sensorKey} - Engine {selectedEngine} - {expandedHistory.length} cycles
              </div>
            </div>
            <div className="detail-panel-actions">
              <button
                className="detail-panel-close"
                type="button"
                onClick={closeModal}
              >
                Close
              </button>
            </div>
          </div>
          <div className="detail-panel-body">
            <div className="detail-panel-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={expandedHistory}
                  margin={{ top: 10, right: 16, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="cycle" tick={{ fill: '#555' }} />
                  <YAxis domain={['dataMin', 'dataMax']} tick={{ fill: '#555' }} />
                  <Tooltip
                    formatter={(value) =>
                      typeof value === 'number' ? value.toFixed(2) : value
                    }
                    labelFormatter={(label) => `Cycle ${label}`}
                  />
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
            <div className="detail-panel-metrics">
              <div className="detail-panel-metric">
                <div className="detail-panel-metric-label">Latest</div>
                <div className="detail-panel-metric-value">
                  {formatMetric(stats?.latest ?? null)}
                </div>
              </div>
              <div className="detail-panel-metric">
                <div className="detail-panel-metric-label">Average</div>
                <div className="detail-panel-metric-value">
                  {formatMetric(stats?.avg ?? null)}
                </div>
              </div>
              <div className="detail-panel-metric">
                <div className="detail-panel-metric-label">Min</div>
                <div className="detail-panel-metric-value">
                  {formatMetric(stats?.min ?? null)}
                </div>
              </div>
              <div className="detail-panel-metric">
                <div className="detail-panel-metric-label">Max</div>
                <div className="detail-panel-metric-value">
                  {formatMetric(stats?.max ?? null)}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '10px',
          marginTop: '20px',
        }}
      >
        {sensors.map((sensor) => (
          <Sparkline
            key={sensor.id}
            label={sensor.label}
            history={sensor.history}
            onClick={() =>
              setActiveSensorKey((prev) =>
                prev === sensor.sensorKey ? null : sensor.sensorKey,
              )
            }
            isActive={sensor.sensorKey === activeSensorKey}
          />
        ))}
      </div>
    </>
  );
}

export default App;
