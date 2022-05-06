
import DemoField from "./DemoField";
import {Typography} from '@mui/material'

function Demo() {
    return (
        <>        
        <Typography variant="h4" component="div" sx={{ flexGrow: 1, paddingTop: "50px", fontWeight: 'bold' }}>
            Demo
        </Typography>
        <DemoField/>
        </>
    );
}

export default Demo;