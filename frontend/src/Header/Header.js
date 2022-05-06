import "./Header.css"
import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
// import Button from '@mui/material/Button';
import { Tabs, Tab } from "@mui/material";
// import { shadows } from '@mui/system';
import {useNavigate, useMatch} from "react-router-dom"

export default function Header() {
  let navigate = useNavigate();
  let match = useMatch('/home/:page');
  const [value, setValue] = React.useState(match.params.page);

  const handleTabChange = (event, newValue) => {
    setValue(newValue);
    navigate(`/home/${newValue}`)
  }
  return (
    <Box sx={{ flexGrow: 1, boxShadow: 2 }} >
      <AppBar position="static" color="inherit">
        <Toolbar>
          <Typography variant="h5" component="div" sx={{ flexGrow: 1 }} align="left" fontWeight={"bold"}>
            Contextual Spelling Correction
          </Typography>          
          <Tabs
            value={value}
            onChange={handleTabChange}
            aria-label="Demo Page"
          >
            <Tab value="demo" label="Demo" />
            <Tab value="about" label="About" />
          </Tabs>

        </Toolbar>
      </AppBar>
    </Box>
  );
}

