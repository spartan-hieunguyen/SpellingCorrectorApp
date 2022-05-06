import React, {useState} from 'react';
import { Box, Button, TextField, Select, MenuItem, InputLabel, FormControl } from '@mui/material';
import './DemoField.css'
import { MODEL } from '../constants';

export default function DemoField() {
    const [input, setInput] = useState('');
    const [output, setOutput] = useState('');
    const [modelOption, setModelOption] = useState(MODEL.TOKEN_CORRECTOR);
    const [hasBeenModified, setHasBeenModified] = useState(true);


    const handleChangeInput = (event) => {
        // logic
        setInput(event.target.value)
        if (!hasBeenModified) {
          setHasBeenModified(true);
          console.log("modifiuing")
        }
    }

    const handleChangeModel = (event) => {
      setModelOption(event.target.value);
    }

    function submit() {
      // const requestOptions = {
      //   method: "POST",
      //   headers: { "Content-Type": "application/json" },
      //   body: JSON.stringify({ text: input }),
      // };
      // // console.log(text)
      // fetch("http://0.0.0.0:8000/correct", requestOptions)
      //   .then((response) => response.json())
      //   .then((data) => {
      //     console.log(data)
      //     setOutput(data)}
      //     )
      //   .catch((error) => {
      //     console.log(error)
      //   })
      if (!hasBeenModified) return;
      setOutput(input);
      setHasBeenModified(false);
      console.log("submiting")
    }
  
    return (
        <Box sx={{display: "flex", flexWrap: "wrap", justifyContent:"center", alignItems:"center", columnGap: "10px", rowGap: "20px", paddingTop: "50px" }}>
          <Box sx={{width: "40%", height:"100%", minWidth:"450px"}}>
              <TextField
              variant="outlined"
              multiline={true}
              minRows={16}
              maxRows={16}
              value={input}
              onChange={handleChangeInput}
              fullWidth
              label="Incorrect"
              InputLabelProps={{shrink: true}}
              InputProps={{ style: { fontSize: "1.2rem" } }}
            />
            <div style={{height: "10px"}}></div>
            <FormControl  fullWidth>
            <InputLabel id="select-model" sx={{shrink: true}}>Model</InputLabel>
            <Select
              labelId="select-model"
              id="select-model"
              value={modelOption}
              defaultValue={MODEL.TOKEN_CORRECTOR}
              label="Model"
              onChange={handleChangeModel}
              fullWidth
              >
              <MenuItem value={MODEL.CORRECTOR}>{MODEL.CORRECTOR}</MenuItem>
              <MenuItem value={MODEL.TOKEN}>{MODEL.TOKEN}</MenuItem>
              <MenuItem value={MODEL.TOKEN_CORRECTOR}>{MODEL.TOKEN_CORRECTOR}</MenuItem>
            </Select>
            </FormControl>
            <div style={{height: "10px"}}></div>
            <Button fullWidth variant="contained" onClick={submit}>Check</Button>
          </Box>
          <Box  sx={{width: "40%", minWidth:"450px"}}>
            <TextField
                  variant="outlined"
                  multiline={true}
                  minRows={20}
                  maxRows={20}
                  value={output}
                  fullWidth 
                  disabled
                  InputLabelProps={{shrink: true}}
                  InputProps={{disableUnderline: true, style: {fontSize: "1.2rem"}}}
                  sx={{
                    "& .MuiInputBase-input.Mui-disabled": {
                      WebkitTextFillColor: "black",
                    },
                  }}
                  label={modelOption}
                />   
          </Box>

        </Box>
    )
}
