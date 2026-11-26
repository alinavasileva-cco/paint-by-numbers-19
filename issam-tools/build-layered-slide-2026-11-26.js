// Layered 4:3 Google Slides builder. Input coordinates use a 1408×1056 reference canvas.
// The caller supplies individual private image paths and text/shape geometry.
function buildLayeredRequests(page, assets, texts, rects, oldBackground) {
  const scale=720/1408, dim=n=>({magnitude:n*scale,unit:"PT"});
  const props=(x,y,w,h)=>({pageObjectId:page,size:{width:dim(w),height:dim(h)},transform:{scaleX:1,scaleY:1,translateX:x*scale,translateY:y*scale,unit:"PT"}});
  const rgb=hex=>({red:parseInt(hex.slice(0,2),16)/255,green:parseInt(hex.slice(2,4),16)/255,blue:parseInt(hex.slice(4,6),16)/255});
  const requests=oldBackground?[{deleteObject:{objectId:oldBackground}}]:[];
  for(const [name,path,x,y,w,h] of assets) requests.push({createImage:{objectId:"issam_asset_"+name,url:path,elementProperties:props(x,y,w,h)}},{updatePageElementAltText:{objectId:"issam_asset_"+name,title:name,description:name}});
  for(const [name,x,y,w,h,hex] of rects) {const id="issam_shape_"+name;requests.push({createShape:{objectId:id,shapeType:"RECTANGLE",elementProperties:props(x,y,w,h)}},{updateShapeProperties:{objectId:id,shapeProperties:{shapeBackgroundFill:{solidFill:{color:{rgbColor:rgb(hex)},alpha:1}},outline:{propertyState:"NOT_RENDERED"},shadow:{propertyState:"NOT_RENDERED"}},fields:"shapeBackgroundFill,outline,shadow"}})}
  for(let i=0;i<texts.length;i++){const [t,x,y,w,h,fs,bold]=texts[i],id="issam_text_"+String(i+1).padStart(2,"0");requests.push({createShape:{objectId:id,shapeType:"TEXT_BOX",elementProperties:props(x,y-3,w,h+8)}},{updateShapeProperties:{objectId:id,shapeProperties:{contentAlignment:"TOP",autofit:{autofitType:"NONE"},insetLeft:{magnitude:0,unit:"PT"},insetRight:{magnitude:0,unit:"PT"},insetTop:{magnitude:0,unit:"PT"},insetBottom:{magnitude:0,unit:"PT"}},fields:"contentAlignment,autofit,insetLeft,insetRight,insetTop,insetBottom"}},{insertText:{objectId:id,text:t}},{updateTextStyle:{objectId:id,textRange:{type:"ALL"},style:{fontFamily:"Arial",fontSize:dim(fs),bold:!!bold,foregroundColor:{opaqueColor:{rgbColor:rgb("061B29")}}},fields:"fontFamily,fontSize,bold,foregroundColor"}},{updateParagraphStyle:{objectId:id,textRange:{type:"ALL"},style:{lineSpacing:100,spaceAbove:{magnitude:0,unit:"PT"},spaceBelow:{magnitude:0,unit:"PT"},alignment:"START"},fields:"lineSpacing,spaceAbove,spaceBelow,alignment"}})}
  return {requests,image_uris:assets.map(a=>a[1])};
}
module.exports = {buildLayeredRequests};
