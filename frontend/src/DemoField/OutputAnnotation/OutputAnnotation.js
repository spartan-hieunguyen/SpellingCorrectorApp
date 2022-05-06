import "./OutputAnnotation.css";
import React, {useEffect, useState} from "react";

function OutputAnnotation({text, spans}) {
  const [output, setOutput] = useState(<></>)
  const createJSXfromNE = (text, spans) => {
      let listOfJSX = [];
      let offset = 0;

      spans.forEach(({start, end}) => {
        const entity = text.slice(start, end);
        const fragments = text.slice(offset, start).split('\n');
        fragments.forEach((fragment, i) => {
          listOfJSX.push(React.createElement('text', null, fragment));
          if (fragments.length > 1 && i !== fragments.length - 1) 
            listOfJSX.push(React.createElement('br'));
        });
        const mark = React.createElement(
          'span',
          {'class': "error__props"},
          React.createElement('text', null, entity));
        listOfJSX.push(mark);
        offset = end;
      });

      listOfJSX.push(React.createElement('text', null, text.slice(offset, text.length)));

      return <>{listOfJSX}</>
  }
  
  useEffect(() => {
    if (text && text.length !== 0 && spans && spans.length !== 0) {
      console.log(text)
      const newAnnotation = createJSXfromNE(text, spans);
      setOutput(newAnnotation);
      return;
    }
    setOutput(<text>{text}</text>)
  }, [text, spans])

  return (
    <div className={"output__box"}>
    {output}
    </div>
  )
}

export default OutputAnnotation;