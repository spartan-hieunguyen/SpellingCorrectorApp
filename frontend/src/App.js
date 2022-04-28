import './App.css';
import Header from './Header';
import DemoField from './DemoField';
import {Typography} from '@mui/material'
function App() {
  return (
    <div className="App">
      <div className="App__header">
        <Header></Header>
      </div>
      <div className="App__content">
        <Typography variant="h5" component="div" sx={{ flexGrow: 1, paddingTop: "50px", fontWeight: 'bold' }}>
            Demo
        </Typography>

        <DemoField/>
      </div>
    </div>
  );
}

export default App;
