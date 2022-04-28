import "./Header.css"
import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import { shadows } from '@mui/system';

export default function Header() {
  return (
    <Box sx={{ flexGrow: 1, boxShadow: 2 }} >
      <AppBar position="static" color="inherit">
        <Toolbar>
          <Typography variant="h5" component="div" sx={{ flexGrow: 1 }} align="left" fontWeight={"bold"}>
            Contextual Spelling Correction
          </Typography>
          <Button color="inherit">Demo</Button>
          <Button color="inherit">About</Button>
        </Toolbar>
      </AppBar>
    </Box>
  );
}

