import './DemoField.css'
import { MODEL_INDEX, MODEL_NAME } from '../../constants';
import React, {useCallback, useState} from 'react';
import OutputAnnotation from './OutputAnnotation/OutputAnnotation';
import { Box, Button, TextField, Select, MenuItem, InputLabel, FormControl, Slider, Typography, Grid, Input } from '@mui/material';

function valuetext(value) {
  return `${value}°C`;
}

export default function DemoField() {
    const [input, setInput] = useState('');
    const [output, setOutput] = useState({text: '', align: []});
    const [modelOption, setModelOption] = useState(MODEL_INDEX.TOKEN_CORRECTOR);
    const [insertPenalty, setInsertPenalty] = useState(4);
    const [deletePenalty, setDeletePenalty] = useState(4);

    const handleChangeInput = useCallback((event) => {
        setInput(event.target.value);
    },[]);

    const handleChangeModel = useCallback((event) => {
      setModelOption(event.target.value);
    },[]);

    const handleInsertSliderChange = (event, newValue) => {
      setInsertPenalty(newValue);
    };
    const handleDeleteSliderChange = (event, newValue) => {
      setDeletePenalty(newValue);
    };
    const handleInsertInputChange = (event) => {
      setInsertPenalty(event.target.value === '' ? 0 : Number(event.target.value));
    }
    const handleDeleteInputChange = (event) => {
      setDeletePenalty(event.target.value === '' ? 0 : Number(event.target.value));
    }
    const handleInsertBlur = () => {
      if (insertPenalty < 0) {
        setInsertPenalty(0);
      } else if (insertPenalty > 20) {
        setInsertPenalty(20);
      }
    }
    const handleDeleteBlur = () => {
      if (deletePenalty < 0) {
        setDeletePenalty(0);
      } else if (deletePenalty > 20) {
        setDeletePenalty(20);
      }
    }
    function submit() {
      const requestOptions = {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          'Access-Control-Allow-Origin': '*'},
        body: JSON.stringify({ 
          text: input, 
          model: modelOption,
          insertPenalty: insertPenalty,
          deletePenalty: deletePenalty,
        }),
      };
      fetch("http://127.0.0.1:8000/correct", requestOptions)
        .then((response) => response.json())
        .then((data) => {
          console.log(data);
          setOutput({text: data.result.text, align: data.result.align})
        })
        .catch((error) => {
          console.log(error);
        });
    }
  
    return (
        <Box sx={{display: "flex", flexWrap: "wrap", justifyContent:"center", alignItems:"center", columnGap: "10px", rowGap: "30px", paddingTop: "30px"}}>
          <Box sx={{width: "35%", height:"560px", minWidth:"450px", display: "flex", flexDirection:"column", justifyContent:"space-between"}}>
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

          <Box sx={{width: "35%", minWidth:"450px", height:"560px", border:"1px solid silver", borderRadius:"5px"}} label={modelOption}>
            <OutputAnnotation
            text={output.text}
            spans={output.align}
            ></OutputAnnotation>
          </Box>
          <Box sx={{display: "flex", width: "80%" ,justifyContent:"center",columnGap: "30px"}}>
            <Box sx={{width: "40%"}}>
            <Typography id="input-slider" gutterBottom>
              Insert Penalty
            </Typography>
            <Grid container spacing={2} alignItems="center">
            <Grid item xs>
              <Slider
                value={insertPenalty}
                onChange={handleInsertSliderChange}
                aria-labelledby="input-slider"
                step={0.5}
                marks
                min={0}
                max={20}
              />
            </Grid>
            <Grid item>
              <Input
                value={insertPenalty}
                size="small"
                onChange={handleInsertInputChange}
                onBlur={handleInsertBlur}
                inputProps={{
                  step: 0.5,
                  min: 0,
                  max: 20,
                  type: 'number',
                  'aria-labelledby': 'input-slider',
                }}
              />
            </Grid>
            </Grid>
            </Box>

            <Box sx={{width: "40%"}}>
            <Typography id="input-slider" gutterBottom>
              Delete Penalty
            </Typography>
            <Grid container spacing={2} alignItems="center">
            <Grid item xs>
              <Slider
                value={deletePenalty}
                onChange={handleDeleteSliderChange}
                aria-labelledby="input-slider"
                step={0.5}
                marks
                min={0}
                max={20}
              />
            </Grid>
            <Grid item>
              <Input
                value={deletePenalty}
                size="small"
                onChange={handleDeleteInputChange}
                onBlur={handleDeleteBlur}
                inputProps={{
                  step: 0.5,
                  min: 0,
                  max: 20,
                  type: 'number',
                  'aria-labelledby': 'input-slider',
                }}
              />
            </Grid>
            </Grid>
            </Box>
          </Box>
        </Box>
    );
}
