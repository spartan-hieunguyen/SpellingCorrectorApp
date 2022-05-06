import './App.css';
// import { useState } from 'react';
// import Header from './Header';
// import DemoField from './Demo/DemoField';
// import {Typography} from '@mui/material'
import Demo from "./Demo"
import About from "./About"
import { Route, Routes, Navigate } from "react-router-dom";
import Home from './Home';

function App() {
  return (
      <Routes>
        <Route  path="/home" element={<Navigate to="/home/demo" />}  />
        <Route  path="*" element={<Navigate to="/home/demo" />}  />
        <Route path="/home" element={<Home/>}>
          <Route path="demo" element={<Demo/>} />
          <Route path="about" element={<About/>} />
        </Route>
      </Routes>
  );
}

export default App;
