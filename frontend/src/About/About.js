import {Typography, Box} from '@mui/material'
import unigram from "../image/model/unigram.png"
import bilabel from "../image/model/bilabel.png"
import bert_ft from "../image/model/bert_ft.png"
import pipeline from "../image/model/pipeline.png"
import result_vsec from "../image/model/result_vsec.png"
import result_viwiki from "../image/model/result_viwiki.png"
import result_augment from "../image/model/result_augment.png"

function About() {
    return <>
        {/* <Typography variant="h5" component="div" sx={{ flexGrow: 1, paddingTop: "50px", fontWeight: 'bold' }}>
Abstract and Motivation</Typography> */}
        <Typography variant="h5" component="div" sx={{ flexGrow: 1, paddingTop: "50px", fontWeight: 'bold' }}>
        Abstract and Motivation
        </Typography>
        <Box sx={{paddingLeft: "15%", paddingRight: "15%", paddingTop: "20px"}}>
        <Typography paragraph align="left">
        Spelling errors are common in everyday life, from writing emails to editing documents. The spelling correction problem consists of two tasks, namely detection and correction. Detecting task can be difficult for real-word errors, when the word appears in the dictionary, however used incorrectly, to which we have to rely on the surrounding context to reason and deduce the correct token. <br/> <br/> Historically, rule-based methods are used to correct misspellings, which does not capture real-word errors and handle more complex misspelling cases, leading to studies of statistical models like N-gram. These model are trained on large corpus, and can capture information using the surrounding text. Nonetheless, a major setback is when two (or more) spelling mistakes stand near each other, the model will not be able to understand and capture the context correctly. Recently, two research group from Vietnam National University and Zalo Corporation propose contextual spelling correction using a deep-learning based approach. <br/> <br/> In this paper, we study the effect of leveraging a pre-trained Transformer Encoder such as phoBERT for Contextual Vietnamese Spelling Correction. We also adapt a tokenization repair module for Vietnamese, to handle merged syllable and splited tokens scenario. With both model combined, we achieved mixed results on two public testing set by Vietnamese National University and Zalo Corporation.
        </Typography>
        </Box>
        <Typography variant="h5" component="div" sx={{ flexGrow: 1, paddingTop: "50px", fontWeight: 'bold' }}>
        Tokenization Repair Model
        </Typography>
        <Typography paragraph align="left" >
        <Box sx={{paddingLeft: "15%", paddingRight: "15%", paddingTop: "20px", display:"flex", flexDirection:"column", justifyContent: "center", rowGap: "30px", alignItems:"center"}}>
        <img src={unigram} alt="" />
        <img src={bilabel} alt="" />

        </Box>
        </Typography>
        <Typography variant="h5" component="div" sx={{ flexGrow: 1, paddingTop: "50px", fontWeight: 'bold' }}>
        Correction Model
        </Typography>
        <Typography paragraph align="left" >
        <Box sx={{paddingLeft: "15%", paddingRight: "15%", paddingTop: "20px", display:"flex", flexDirection:"column", justifyContent: "center", rowGap: "30px", alignItems:"center"}}>
        <img src={bert_ft} alt="" />
        </Box>
        </Typography>
        <Typography variant="h5" component="div" sx={{ flexGrow: 1, paddingTop: "50px", fontWeight: 'bold' }}>
        Full Pipeline
        </Typography>
        <Typography paragraph align="left" >
        <Box sx={{paddingLeft: "15%", paddingRight: "15%", paddingTop: "20px", display:"flex", flexDirection:"column", justifyContent: "center", rowGap: "30px", alignItems:"center"}}>
        <img src={pipeline} alt="" />
        </Box>
        </Typography>
        <Typography variant="h5" component="div" sx={{ flexGrow: 1, paddingTop: "50px", fontWeight: 'bold' }}>
        Result
        </Typography>
        <Typography paragraph align="left" >
        <Box sx={{paddingLeft: "15%", paddingRight: "15%", paddingTop: "20px", display:"flex", flexDirection:"column", justifyContent: "center", rowGap: "30px", alignItems:"center"}}>
        <img src={result_augment} alt="" style={{width: "600px"}}/>
        <img src={result_vsec} alt="" style={{width: "600px"}}/>
        <img src={result_viwiki} alt="" style={{width: "600px"}}/>
        </Box>
        </Typography>
        <Typography variant="h5" component="div" sx={{ flexGrow: 1, paddingTop: "50px", fontWeight: 'bold' }}>
        References
        </Typography>
        <Typography paragraph align="left" >
        <Box sx={{paddingLeft: "15%", paddingRight: "15%", paddingTop: "20px"}}>
        </Box>
        </Typography>
    </>
}

export default About;