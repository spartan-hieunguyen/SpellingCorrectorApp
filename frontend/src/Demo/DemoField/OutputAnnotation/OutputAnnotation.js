import "./OutputAnnotation.css";
import React, {useEffect, useState, useCallback} from "react";

function OutputAnnotation({text, spans}) {
  const [output, setOutput] = useState(<></>)
  
  const createReactText = useCallback((text) => {
    return React.createElement('text', null, text.replace(/ /g, "\u00a0"));
  },[])

  const trimAndAddNewline = useCallback((strings) => {
    let ans = [];
    let strs = strings.split('\n');
    strs.forEach((s, i) => {
      if (s.length > 0)
        ans.push(createReactText(s));
      if (strs.length > 1 && i !== s.length - 1) 
        ans.push(React.createElement('br'));
    })
    return ans;
  },[createReactText])

  const annotateText = useCallback((text, spans) => {
      let listOfJSX = [];
      let offset = 0;

      spans.forEach(({start, end}) => {
        const entity = text.slice(start, end);
        let fragments = trimAndAddNewline(text.slice(offset, start));
        listOfJSX.push(...fragments);
        const mark = React.createElement(
          'span',
          {'className': "error__props"},
          createReactText(entity));
        listOfJSX.push(mark);
        offset = end;
      });

      const remains = trimAndAddNewline(text.slice(offset, text.length));
      listOfJSX.push(...remains);
      return <>{listOfJSX}</>
  }, [createReactText,trimAndAddNewline])

  useEffect(() => {
    if (text && text.length !== 0 && spans && spans.length !== 0) {
      setOutput(annotateText(text, spans));
      return;
    }
    setOutput(text)
  }, [text, spans, annotateText])

  return (
    <div className={"output__box"}>
    {output}
    </div>
  )
}

export default OutputAnnotation;