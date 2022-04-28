import React, {useState} from 'react';
import { Box, Button, TextField } from '@mui/material';
import './DemoField.css'

export default function DemoField() {
    const [input, setInput] = useState('');
    const [output, setOutput] = useState('');


    const handleChangeInput = (event) => {
        // logic
        setInput(event.target.value)
    }

    function submit() {
      const requestOptions = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      };
      // console.log(text)
      fetch("http://0.0.0.0:8000/correct", requestOptions)
        .then((response) => response.json())
        .then((data) => {
          console.log(data)
          setOutput(data)}
          )
        .catch((error) => {
          console.log(error)
        })
    }
  
    return (
        <Box sx={{display: "flex", flexWrap: "wrap", justifyContent:"center", alignItems:"center", columnGap: "10px", rowGap: "20px", paddingTop: "50px" }}>
          <Box sx={{width: "40%", height:"100%", minWidth:"450px"}}>
              <TextField
              variant="outlined"
              multiline={true}
              rows={18}
              rowsMax={18}
              value={input}
              onChange={handleChangeInput}
              fullWidth
              label="Incorrect"
              InputLabelProps={{shrink: true}}
            />
            <div style={{height: "10px"}}></div>
            <Button fullWidth variant="contained" onSubmit={submit}>Check</Button>
          </Box>
          <Box  sx={{width: "40%", minWidth:"450px"}}>
            <TextField
                  variant="outlined"
                  multiline={true}
                  rows={20}
                  rowsMax={20}
                  value={output}
                  fullWidth 
                  disabled
                  InputLabelProps={{shrink: true}}
                  InputProps={{disableUnderline: true}}
                  sx={{
                    "& .MuiInputBase-input.Mui-disabled": {
                      WebkitTextFillColor: "black",
                    },
                  }}
                  label="Correct"
                />   
          </Box>

        </Box>
    )
}
