import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const root = path.resolve(".");
const skill = process.env.PRESENTATION_SKILL_DIR ?? path.join(os.homedir(), ".codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations");
const python = process.env.ARTIFACT_RUNTIME_PYTHON ?? path.join(os.homedir(), ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3");
process.env.RUNTIME_NODE_MODULES ??= path.join(os.homedir(), ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules");
const team = JSON.parse(await fs.readFile("configs/team.json", "utf8"));
const content = JSON.parse(await fs.readFile("build/midterm_content.json","utf8"));
const presentation = Presentation.create({slideSize:{width:1280,height:720}});
const ink="#163139", teal="#087e75", muted="#52666d", pale="#edf6f4";

function text(slide,value,x,y,w,h,size=28,color=ink,bold=false,align="left"){
  const shape=slide.shapes.add({geometry:"textbox",position:{left:x,top:y,width:w,height:h},
    fill:"none",line:{fill:"none",width:0}});
  shape.text=value;
  shape.text.style={typeface:"Arial",fontSize:size,color,bold,alignment:align,
    verticalAlignment:"top",wrap:"square",insets:0};
  return shape;
}
function page(title,index){
  const slide=presentation.slides.add();
  slide.background.fill="#ffffff";
  text(slide,title,64,48,1152,70,44,ink,true);
  text(slide,`${index} / 6`,1135,672,80,24,17,muted,false,"right");
  return slide;
}
function table(slide,values,x,y,width,height,widths,size=26){
  const t=slide.tables.add({rows:values.length,columns:values[0].length,
    left:x,top:y,width,height,columnWidths:widths,values});
  t.borders.assign({style:"solid",fill:"#d7e3e2",width:1});
  for(let i=0;i<values.length;i++) for(let j=0;j<values[0].length;j++){
    const c=t.getCell(i,j);
    c.fill=i===0?ink:(i%2===1?"#f2f7f6":"#ffffff");
    c.text.style={typeface:"Arial",fontSize:size,color:i===0?"#ffffff":ink,
      bold:i===0,alignment:j===0?"left":"center",verticalAlignment:"middle",
      insets:12};
  }
  return t;
}
const src=[
 "Motivating sentences are authored examples, not benchmark observations. BERT: https://aclanthology.org/N19-1423/",
 "BioScope official annotations: https://rgai.inf.u-szeged.hu/node/105 ; data audit: results/metrics/pilot_data_audit.json. Sentence labels are annotation-derived cue presence.",
 "Code: src/data_loader.py, src/feature_extraction.py, src/classifiers.py. Checkpoint revisions: configs/models.json. BioBERT: https://doi.org/10.1093/bioinformatics/btz682 ; PubMedBERT: https://doi.org/10.1145/3458754",
 "Measured results: results/metrics/pilot/summary.csv and individual run/prediction JSON. Integrity audit checked all 75 evaluations. Fixed test N=600, validation N=400. Full pool changes class distribution and is excluded from balanced learning-curve lines.",
 "Limitations from actual task/protocol. BioScope: https://rgai.inf.u-szeged.hu/node/105 . Potential pretraining overlap is not ruled out by document-group downstream splitting.",
 "Future work only. SemEval target annotations and access requirements: https://physionet.org/content/semeval2015/2.0/ . Synthetic diagnostics: data/minimal_pairs_demo.json."
];

// 1. Minimal title plus the motivating contrast as editable text.
{
  const s=page("Probing clinical meaning",1);
  text(s,"Negation, uncertainty, and terminology generalization",64,124,1152,60,30,muted);
  text(s,`CSI 436/536 · Project P · Group ${team.group_id}`,64,190,1152,36,22,muted);
  text(s,team.members.join(" · "),64,232,1152,36,24,ink,true);
  text(s,"The patient has pneumonia.",88,292,1080,54,37,ink);
  text(s,"The patient has no pneumonia.",88,370,1080,54,37,teal,true);
  text(s,"The patient may have pneumonia.",88,448,1080,54,37,teal,true);
  text(s,"Can frozen text embeddings help a simple classifier detect these distinctions?",64,569,1135,78,29,ink,true);
}
// 2. Schema is explicit, with the broader questions clearly scoped.
{
  const s=page("Problem setup and research questions",2);
  text(s,"Current task: sentence-level cue presence in BioScope abstracts",64,133,1140,45,28,teal,true);
  table(s,[["Label","Definition","Retained sentences"],
    ["No cue","Neither annotated cue type","8,452"],
    ["Negation only","Negation cue, no speculation cue","1,301"],
    ["Speculation only","Speculation cue, no negation cue","1,806"]],
    64,200,1152,248,[270,580,302],25);
  text(s,"RQ1  Which frozen features support this cue-detection task?",64,479,1152,40,27);
  text(s,"RQ2  How do results change with fewer training labels?",64,530,1152,40,27);
  text(s,"RQ3  Unseen diseases require additional target-level annotations",64,581,1152,55,27,muted);
}
// 3. Explicitly requested editable method diagram.
{
  const s=page("Proposed method",3);
  const labels=["BioScope text\nDocument-group split","Features\nTF-IDF or frozen encoder",
                "Linear classifier\nLR or linear SVM","Evaluation\nMacro-F1 and recall"];
  const nodes=labels.map((label,i)=>{
    const sh=s.shapes.add({geometry:"rect",position:{left:64+i*298,top:182,width:258,height:128},
      fill:pale,line:{fill:"#87bcb4",width:1.2}});
    sh.text=label;
    sh.text.style={typeface:"Arial",fontSize:25,color:ink,alignment:"center",
      verticalAlignment:"middle",insets:10};
    return sh;
  });
  for(let i=0;i<3;i++)s.shapes.connect(nodes[i],nodes[i+1],{
    kind:"straight",fromSide:"right",toSide:"left",
    line:{fill:teal,width:2},tail:{type:"arrow",width:"med",length:"med"}});
  text(s,"BERT-base-cased, BioBERT v1.2, and PubMedBERT",64,352,1152,44,29,teal,true);
  text(s,"Frozen last-layer mean pooling; padding and special tokens excluded",64,402,1152,44,26);
  text(s,"Pilot: 1,500 train / 400 validation / 600 test sentences",64,466,1152,44,29,ink,true);
  text(s,"10, 25, 50, 100 examples per class and the full pilot training pool",64,516,1152,42,26);
  text(s,"Same training IDs for every method; C selected on validation; 3 sampling seeds",64,568,1152,60,25,muted);
}
// 4. Native numeric table backed by saved metrics.
{
  const s=page("Current progress: completed pilot",4);
  const pct=(pair,sd=true)=>`${(pair[0]*100).toFixed(1)}${sd?` ± ${(pair[1]*100).toFixed(1)}`:""}`;
  const values=[["Method","10 / class","100 / class","Full pool"]];
  content.methods.forEach(m=>values.push([m.label,pct(m.ten),pct(m.hundred),pct(m.full,false)]));
  text(s,"Test macro-F1 (%)",64,132,1152,42,29,teal,true);
  table(s,values,64,190,1152,300,[390,254,254,254],27);
  text(s,"75 evaluations completed; all three encoder caches and raw predictions saved",64,516,1152,50,26,ink,true);
  text(s,"Mean ± sample SD across training subsets on one fixed test set",64,568,1152,40,23,muted);
  text(s,"Full pool: 1,500 naturally imbalanced examples; different class mix from balanced budgets",64,611,1152,43,22,muted);
}
// 5. Paired issues and implications.
{
  const s=page("Challenges and limits",5);
  const rows=[
    ["Cue words and scope","Sentence labels cannot establish the status of a particular disease."],
    ["Class imbalance","439 of 600 test sentences have no cue. Macro-F1 keeps minority classes visible."],
    ["Label budget and uncertainty","Every run also uses 400 validation labels. SDs describe training-subset variation."],
    ["Generalization","Near-duplicate templates and unsupervised pretraining overlap need further audits."]
  ];
  rows.forEach((r,i)=>{
    text(s,r[0],64,159+i*120,1152,42,29,teal,true);
    text(s,r[1],64,207+i*120,1120,60,26,ink);
  });
}
// 6. Direct next steps without claiming the unrun extensions.
{
  const s=page("Future plans",6);
  text(s,"01",64,174,80,70,41,teal,true);
  text(s,"Expand the corpus experiment",164,169,1010,48,32,ink,true);
  text(s,"Use the uncapped split and inspect errors by cue family",164,223,1010,60,27,muted);
  text(s,"02",64,330,80,70,41,teal,true);
  text(s,"Retain mixed cues",164,325,1010,48,32,ink,true);
  text(s,"Evaluate two binary labels and controlled sentence diagnostics",164,379,1010,60,27,muted);
  text(s,"03",64,486,80,70,41,teal,true);
  text(s,"Add target-level evidence",164,481,1010,48,32,ink,true);
  text(s,"Obtain reliable entity labels before span pooling or unseen-disease tests",164,535,1010,78,27,muted);
}
for(let i=0;i<presentation.slides.items.length;i++){
  presentation.slides.items[i].speakerNotes.text=content.scripts[i].text+"\n\nSources and scope:\n"+src[i];
}
await fs.mkdir("build/slide-previews",{recursive:true});
await fs.mkdir("build/finalization",{recursive:true});
const candidate=path.join(root,"build/midterm_candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidate);
for(const [i,slide] of presentation.slides.items.entries()){
  const blob=await presentation.export({slide,format:"png",scale:1});
  await fs.writeFile(`build/slide-previews/slide-${i+1}.png`,new Uint8Array(await blob.arrayBuffer()));
}
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,"container_tools/artifact_tool_utils.mjs")).href);
const output=path.resolve(process.env.MIDTERM_PPTX ?? "slides/midterm_presentation.pptx");
const result=await finalizePresentation({
  workspaceDir:root,candidatePath:candidate,finalPath:output,
  explicitTotalSlideCount:6,requiredNativeTableOwnerSlides:[2,4],
  requiredNativeChartOwnerSlides:[],
  pythonExecutable:python,
  integrityValidatorPath:path.join(skill,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(skill,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-bullet-geometry",
              "--validate-heading-fit","--require-native-table-slide","2","--require-native-table-slide","4"],
  fontPolicy:{basis:"design",families:["Arial"]},
  verifyArtifactToolImport:true,
  receiptPath:path.join(root,"build/finalization",`${path.basename(output,".pptx")}.validation.json`)
});
console.log(JSON.stringify(result));
