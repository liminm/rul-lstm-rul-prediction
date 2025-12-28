import { useState, useEffect } from 'react'
import SensorDisplay from './components/SensorDisplay'

function App() {
  const [count, setCount] = useState(0)

  const fetchData = async () => {
    try {
      console.log("Fetching data from Python...");
      
      // 1. Make the request
      const response = await fetch("http://127.0.0.1:8001/predict/", {
        method: "POST", // Your endpoint expects a POST
        headers: {
          "Content-Type": "application/json",
        },
        // We send an empty body for now since your backend is hardcoded to read the file
        body: JSON.stringify({}), 
      });
      
      // 2. Parse the JSON answer
      //const data = await response.json();


      const data = [
        { id: 1, label: "Fan Speed", value: 123 },
        { id: 2, label: "Core Temp", value: 6545 },
        { id: 3, label: "Oil Pressure", value: 812120 },
      ]

      console.log("Response received:", data);
      // 3. Update the state with the prediction
      // Your API returns { "predicted_rul": 145.3 }
      setSensors(data);
      //setCount(data.predicted_rul);
      
    } catch (error) {
      console.error("Error fetching data:", error);
    };
  };

  useEffect(() => {
      // Call the function immediately
      fetchData();
    }, []);

  const [sensors, setSensors] = useState([
    { id: 1, label: "Fan Speed", value: 2500 },
    { id: 2, label: "Core Temp", value: 540 },
    { id: 3, label: "Oil Pressure", value: 80 },
  ]);



  return (
    <>
      <h1>Engine Dashboard</h1>
      <div className="card">
        <SensorDisplay value={count} label="Sensor Count" />
        <button onClick={() => fetchData()}>
          Update Sensor
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      {/* 2. Map over the 'sensors' state variable now */}
      {sensors.map((sensor) => (
        <SensorDisplay value={sensor.value} label={sensor.label} key={sensor.id} />
      ))}
    </>
  )
}

export default App

