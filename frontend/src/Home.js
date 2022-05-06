import Header from "./Header"
import { Outlet } from "react-router-dom";

function Home() {
    return (
    <div className="App">
      <div className="App__header">
        <Header/>
      </div>
      <div className="App__content">
        <Outlet/>
        <div style={{minHeight: "20px"}}></div>
      </div>
    </div>
    )
}

export default Home;