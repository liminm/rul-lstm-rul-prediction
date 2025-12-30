import { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Brush,
  ResponsiveContainer,
  CartesianGrid,
  Tooltip,
} from 'recharts';
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

function App() {
  const [windowRange, setWindowRange] = useState({ startIndex: 0, endIndex: 50 });
  const [fullHistory, setFullHistory] = useState([]);
  const [selectedEngine, setSelectedEngine] = useState(1);
  const [availableEngines, setAvailableEngines] = useState<number[]>([]);
  const [rulPrediction, setRulPrediction] = useState<number | null>(null);
  const [sensors, setSensors] = useState([{ id: 1, label: 'Loading...', value: 0, history: [] }]);

  useEffect(() => {
    const fetchEngines = async () => {
      try {
        const response = await fetch('http://localhost:8001/engines/');
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
    const fetchRulPrediction = async () => {
      try {
        const response = await fetch('http://localhost:8001/predict/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            unit_nr: selectedEngine,
          }),
        });
        const data = await response.json();
        setRulPrediction(data.predicted_rul);
      } catch (error) {
        console.error('Error fetching RUL prediction:', error);
      }
    };

    fetchRulPrediction();
  }, [selectedEngine]);

  const fetchData = async () => {
    try {
      const response = await fetch(
        `http://127.0.0.1:8001/sensors/?start_cycle=${windowRange.startIndex}&end_cycle=${windowRange.endIndex}&limit=500&unit_nr=${selectedEngine}`,
      );
      const data = await response.json();

      const latestReading = data[0];
      console.log('Latest Reading:', latestReading);

      const formattedSensors = Object.keys(ALL_SENSOR_NAMES).map((key, index) => ({
        id: index,
        label: ALL_SENSOR_NAMES[key as keyof typeof ALL_SENSOR_NAMES],
        history: data.map((row: Record<string, number>) => ({ value: row[key] })),
      }));

      setSensors(formattedSensors);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const fetchFullHistory = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8001/sensors/?limit=500&unit_nr=${selectedEngine}`);
      const data = await response.json();
      console.log('Full History Data:', data);
      setFullHistory(data);
    } catch (err) {
      console.error('Timeline fetch failed:', err);
    }
  };

  useEffect(() => {
    fetchFullHistory();
  }, [selectedEngine]);

  useEffect(() => {
    fetchData();
  }, [windowRange, selectedEngine]);

  return (
    <>
      <h1>Engine Dashboard</h1>
      <div className="card">
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
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '10px',
          marginTop: '20px',
        }}
      >
        {sensors.map((sensor) => (
          <Sparkline key={sensor.id} label={sensor.label} history={sensor.history} />
        ))}
      </div>
    </>
  );
}

export default App;
