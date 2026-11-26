// Layered 4:3 Google Slides builder. Coordinates use a 1408×1056 reference canvas.
// Caller supplies individual private image paths and text/shape geometry.
function buildLayeredRequests(page,assets,texts,rects,oldBackground){
 const k=720/1408,dim=n=>({magnitude:n*k,unit:"PT"}),props=(x,y,w,h)=>({pageObjectId:page,size:{width:dim(w),height:dim(h)},transform:{scaleX:1,scaleY:1,translateX:x*k,translateY:y*k,unit:"PT"}}),rgb=h=>({red:parseInt(h.slice(0,2),16)/255,green:parseInt(h.slice(2,4),16)/255,blue:parseInt(h.slice(4,6),16)/255});
 const requests=oldBackground?[{deleteObject:{objectId:oldBackground}}]:[];
 for(const [n,url,x,y,w,h] of assets)requests.push({createImage:{objectId:"issam_asset_"+n,url,elementProperties:props(x,y,w,h)}},{updatePageElementAltText:{objectId:"issam_asset_"+n,title:n,description:n}});
 for(const [n,x,y,w,h,c] of rects){const id="issam_shape_"+n;requests.push({createShape:{objectId:id,shapeType:"RECTANGLE",elementProperties:props(x,y,w,h)}},{updateShapeProperties:{objectId:id,shapeProperties:{shapeBackgroundFill:{solidFill:{color:{rgbColor:rgb(c)},alpha:1}},outline:{propertyState:"NOT_RENDERED"},shadow:{propertyState:"NOT_RENDERED"}},fields:"shapeBackgroundFill,outline,shadow"}})}
 for(let i=0;i<texts.length;i++){const [t,x,y,w,h,fs,b]=texts[i],id="issam_text_"+String(i+1).padStart(2,"0");requests.push({createShape:{objectId:id,shapeType:"TEXT_BOX",elementProperties:props(x-7,y-10,w+14,h+20)}},{insertText:{objectId:id,text:t}},{updateTextStyle:{objectId:id,textRange:{type:"ALL"},style:{fontFamily:"Arial",fontSize:dim(fs),bold:!!b,foregroundColor:{opaqueColor:{rgbColor:rgb("061B29")}}},fields:"fontFamily,fontSize,bold,foregroundColor"}},{updateParagraphStyle:{objectId:id,textRange:{type:"ALL"},style:{lineSpacing:100,spaceAbove:{magnitude:0,unit:"PT"},spaceBelow:{magnitude:0,unit:"PT"},alignment:"START"},fields:"lineSpacing,spaceAbove,spaceBelow,alignment"}})}
 return {requests,image_uris:assets.map(a=>a[1])};
}
module.exports={buildLayeredRequests};
