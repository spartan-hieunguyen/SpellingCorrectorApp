import './DemoField.css'
import { MODEL_INDEX, MODEL_NAME } from '../../constants';
import React, {useCallback, useState} from 'react';
import OutputAnnotation from './OutputAnnotation/OutputAnnotation';
import { Box, Button, TextField, Select, MenuItem, InputLabel, FormControl } from '@mui/material';

export default function DemoField() {
    const [input, setInput] = useState('');
    const [output, setOutput] = useState('');
    const [modelOption, setModelOption] = useState(MODEL_INDEX.TOKEN_CORRECTOR);
    const [hasBeenModified, setHasBeenModified] = useState(true);


    const handleChangeInput = useCallback((event) => {
        // logic
        setInput(event.target.value);
        if (!hasBeenModified) {
          setHasBeenModified(true);
          // console.log("modifying")
        }
    },[hasBeenModified])

    const handleChangeModel = useCallback((event) => {
      setModelOption(event.target.value);
    },[])

    function submit() {
      const requestOptions = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input, model: modelOption }),
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
      // if (!hasBeenModified) return;
      // setOutput(input.trim());
      // setHasBeenModified(false);
      // console.log(output)
      // console.log("submiting")
    }
  
    return (
        <Box sx={{display: "flex", flexWrap: "wrap", justifyContent:"center", alignItems:"center", columnGap: "10px", rowGap: "30px", paddingTop: "30px"}}>
          <Box sx={{width: "40%", height:"550px", minWidth:"450px", display: "flex", flexDirection:"column", justifyContent:"space-between"}}>
              <TextField
              variant="outlined"
              multiline={true}
              minRows={15}
              maxRows={15}
              value={input}
              onChange={handleChangeInput}
              fullWidth
              label="Incorrect"
              InputLabelProps={{shrink: true}}
              InputProps={{ style: { fontSize: "1.2rem" } }}
            />
            <FormControl  fullWidth>
            <InputLabel id="select-model" sx={{shrink: true}}>Model</InputLabel>
            <Select
              labelId="select-model"
              id="select-model"
              value={modelOption}
              defaultValue={MODEL_INDEX.TOKEN_CORRECTOR}
              label="Model"
              onChange={handleChangeModel}
              fullWidth
              >
              <MenuItem value={MODEL_INDEX.CORRECTOR}>{MODEL_NAME.CORRECTOR}</MenuItem>
              <MenuItem value={MODEL_INDEX.TOKEN}>{MODEL_NAME.TOKEN}</MenuItem>
              <MenuItem value={MODEL_INDEX.TOKEN_CORRECTOR}>{MODEL_NAME.TOKEN_CORRECTOR}</MenuItem>
            </Select>
            </FormControl>
            <Button fullWidth variant="contained" onClick={submit}>Check</Button>
          </Box>

          <Box sx={{width: "40%", minWidth:"450px", height:"550px", border:"1px solid silver", borderRadius:"5px"}} label={modelOption}>
            <OutputAnnotation text={output} spans={[{"start":0,"end":2}]}/>
          </Box>
        </Box>
    )
}
