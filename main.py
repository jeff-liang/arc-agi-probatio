
import os, json, glob, base64, argparse, asyncio, random, re, math, statistics, inspect
from io import BytesIO
from typing import List, Tuple, Any, Dict, Optional
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from PIL import Image, ImageDraw, ImageFont

# ---------- Config ----------
DEFAULT_PALETTE = {
    0:(0,0,0),1:(0,0,255),2:(255,0,0),3:(0,255,0),4:(255,255,0),
    5:(128,128,128),6:(255,0,255),7:(255,165,0),8:(0,128,128),9:(128,0,0)
}
CELL=24; GRID_PAD=6; ARROW_PAD=16; ARROW_WIDTH=28

# ---------- Utilities ----------
def grid_to_text(grid:List[List[int]])->str:
    return "\\n".join(" ".join(str(v) for v in row) for row in grid)

def example_text_pair(inp:List[List[int]], out:List[List[int]])->str:
    return f"Input:\\n{grid_to_text(inp)}\\n\\nOutput:\\n{grid_to_text(out)}"

def question_text(inp:List[List[int]])->str:
    return f"Input:\\n{grid_to_text(inp)}\\n\\nOutput:\\n?"

def render_grid(grid:List[List[int]], palette:Dict[int,tuple]=DEFAULT_PALETTE)->Image.Image:
    H=len(grid); W=len(grid[0])
    img=Image.new("RGB",(W*CELL+2*GRID_PAD,H*CELL+2*GRID_PAD),(255,255,255))
    d=ImageDraw.Draw(img)
    for r in range(H):
        for c in range(W):
            x0=GRID_PAD + c*CELL; y0=GRID_PAD + r*CELL
            x1=x0+CELL-1; y1=y0+CELL-1
            color=palette.get(grid[r][c], (0,0,0))
            d.rectangle([x0,y0,x1,y1], fill=color, outline=(220,220,220))
    d.rectangle([0,0,img.width-1,img.height-1], outline=(200,200,200))
    return img

def render_question_box(h: int, w: int)->Image.Image:
    QUESTION_W=w*CELL; QUESTION_H=h*CELL
    img=Image.new("RGB",(QUESTION_W,QUESTION_H),(255,255,255))
    d=ImageDraw.Draw(img)
    d.rectangle([0,0,img.width-1,img.height-1],outline=(180,180,180))
    try: font=ImageFont.truetype("DejaVuSans.ttf",96)
    except: font=ImageFont.load_default()
    txt="?"
    bbox=d.textbbox((0,0),txt,font=font)
    tw=bbox[2]-bbox[0]; th=bbox[3]-bbox[1]
    d.text(((img.width-tw)//2,(img.height-th)//2),txt,fill=(0,0,0),font=font)
    return img

def compose_lr_arrow(left:Image.Image,right:Image.Image)->Image.Image:
    H=max(left.height,right.height)
    W=left.width+ARROW_PAD+ARROW_WIDTH+ARROW_PAD+right.width
    img=Image.new("RGB",(W,H),(255,255,255)); d=ImageDraw.Draw(img)
    x=0
    img.paste(left,(x,(H-left.height)//2)); x+=left.width+ARROW_PAD
    y_mid=H//2; x0=x; x1=x+ARROW_WIDTH-10
    d.line((x0,y_mid,x1,y_mid),fill=(0,0,0),width=3)
    d.polygon([(x1,y_mid-7),(x1,y_mid+7),(x1+10,y_mid)],fill=(0,0,0))
    x+=ARROW_WIDTH+ARROW_PAD
    img.paste(right,(x,(H-right.height)//2))
    d.rectangle([0,0,img.width-1,img.height-1],outline=(200,200,200))
    return img

def pil_to_data_url(img:Image.Image)->str:
    buf=BytesIO(); img.save(buf,format="PNG")
    b64=base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

def load_arc_tasks(dataset_dir:str, limit:Optional[int]=None)->List[Dict[str,Any]]:
    dataset_path = Path(dataset_dir)
    tasks=[]
    if dataset_path.is_file():
        with open(dataset_path, "r") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected top-level object in dataset file: {dataset_dir}")
        for task_id, task in obj.items():
            tasks.append({
                "task_id": task_id,
                "path": str(dataset_path),
                "task": task,
            })
    else:
        paths=sorted(glob.glob(os.path.join(dataset_dir,"*.json")))
        for p in paths:
            with open(p,"r") as f:
                tasks.append({
                    "task_id": Path(p).stem,
                    "path": p,
                    "task": json.load(f),
                })
    random.shuffle(tasks)
    return tasks[:limit] if limit else tasks

# ---------- Validation helpers ----------
def is_grid(x)->bool:
    return isinstance(x,list) and len(x)>0 and all(isinstance(r,list) and len(r)>0 for r in x) and \
           len({len(r) for r in x})==1 and all(all(isinstance(v,int) and 0<=v<=9 for v in r) for r in x)

def grids_equal(a:List[List[int]], b:List[List[int]])->bool:
    return is_grid(a) and is_grid(b) and len(a)==len(b) and len(a[0])==len(b[0]) and all(a[r][c]==b[r][c] for r in range(len(a)) for c in range(len(a[0])))

def per_cell_accuracy(a:List[List[int]], b:List[List[int]])->float:
    if not (is_grid(a) and is_grid(b) and len(a)==len(b) and len(a[0])==len(b[0])): return 0.0
    H=len(a); W=len(a[0])
    match=sum(1 for r in range(H) for c in range(W) if a[r][c]==b[r][c])
    return match/(H*W)

# ---------- JSON robust extraction ----------
def extract_json_block(text:str)->str:
    if "```json" in text:
        text = text.split("```json")
        text = text[1]
        text = text.split("```")
        return text[0]
    text=text.strip()
    stack=[]
    start=None
    for i,ch in enumerate(text):
        if ch in '{[':
            if start is None:
                start=i
            stack.append(ch)
        elif ch in '}]' and stack:
            stack.pop()
            if not stack:
                end=i+1
                return text[start:end]
    return text

# ---------- Naive grid parser (baseline) ----------
def parse_multiple_grids(text:str, expected_k:int, log_failures:bool=True)->Tuple[List[List[List[int]]], List[List[List[int]]]]:
    block=extract_json_block(text)
    try:
        obj=json.loads(block)
        if isinstance(obj, list):
            first=obj; second=[]
        elif isinstance(obj, dict):
            first=obj.get("first_try", obj.get("answer", obj.get("outputs", [])))
            second=obj.get("second_try", [])
        else:
            first=[]; second=[]
    except Exception:
        first=[]; second=[]
    def coerce_to_k_grids(x):
        if expected_k==1 and is_grid(x):
            return [x]
        if isinstance(x,list):
            grids=[g for g in x if is_grid(g)]
            if len(grids)>=expected_k:
                return grids[:expected_k]
        return ["" for _ in range(expected_k)]
    first_k=coerce_to_k_grids(first)
    second_k=coerce_to_k_grids(second)
    if not second_k:
        second_k=["" for _ in range(expected_k)]
    if log_failures and first_k[0] == "":
        print("First grid empty")
    if log_failures and second_k[0] == "":
        print("Second grid empty")
    return first_k, second_k

# ---------- Prompt builders ----------
def build_messages_for_task_grid(task:Dict[str,Any], mode:str)->Tuple[List[Dict], Optional[List[List[List[int]]]]]:
    train_pairs=task["train"]; test_pairs=task["test"]
    k=len(test_pairs); assert k>=1
    examples_text=[]; example_images=[]
    for ex in train_pairs:
        inp=ex["input"]; out=ex["output"]
        examples_text.append(example_text_pair(inp,out))
        if mode=="multimodal":
            example_images.append(pil_to_data_url(compose_lr_arrow(render_grid(inp),render_grid(out))))
    q_text_lines=["Questions:"]; q_images=[]; golds=[]
    has_golds=True
    for i, ex in enumerate(test_pairs, start=1):
        inp=ex["input"]
        if "output" in ex:
            golds.append(ex["output"])
        else:
            has_golds=False
        q_text_lines.append(f"Q{i}: {question_text(inp)}")
        if mode=="multimodal":
            q_images.append(pil_to_data_url(compose_lr_arrow(render_grid(inp),render_question_box(len(inp), len(inp[0])))))
    system=(
        "You are a puzzle solving expert solving ARC tasks. Each example shows an input grid and its correct output grid. "
        "Grids use digits 0-9 for colors. Infer the transformation and apply it to each question. "
        "Do not assume the output size; it can differ from the input. You get two tries for each grid."
    )
    user_parts=[
        {"type":"text","text":"Examples:"}]
    for i,txt in enumerate(examples_text, start=1):
        user_parts.append({"type":"text","text":f"(Example {i})\\n{txt}"})
        if mode=="multimodal":
            user_parts.append({"type":"image_url","image_url":{"url":example_images[i-1]}})
    user_parts.append({"type":"text","text":"\\n" + "\\n".join(q_text_lines)})
    if mode=="multimodal":
        for i,url in enumerate(q_images, start=1):
            user_parts.append({"type":"text","text":f"(Q{i} image)"})
            user_parts.append({"type":"image_url","image_url":{"url":url}})
    user_parts.append({"type":"text","text":
        """Return in the following format. Example for two question grids:\\n
        <rationale>
        Insert rationale here
        </rationale>
        ```json
        {
          "first_try":[
            [
              [0,1],
              [1,0]
            ],
            [
              [2]
            ]
          ],
          "second_try":[
            [
              [0,0],
              [0,0]
            ],
            [
              [2]
            ]
          ]
        }
        ```
        
        Be super thorough. You can do it!
        Make sure your grids accurately represent the pattern you've discovered!"""
    })
    messages=[{"role":"system","content":system},{"role":"user","content":user_parts}]
    return messages, golds if has_golds else None

def normalize_submission_grid(grid:Any)->List[List[int]]:
    return grid if is_grid(grid) else []

def normalize_model_name(model:str)->str:
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model

def messages_to_openai_input(messages:List[Dict])->Tuple[Optional[str], List[Dict[str,Any]]]:
    instructions_parts=[]
    openai_input=[]
    for message in messages:
        role=message["role"]
        content=message["content"]
        if role == "system":
            if isinstance(content, str):
                instructions_parts.append(content)
            else:
                instructions_parts.extend(part["text"] for part in content if part.get("type") == "text")
            continue
        content_parts=[]
        if isinstance(content, str):
            content_parts.append({"type": "input_text", "text": content})
        else:
            for part in content:
                if part["type"] == "text":
                    content_parts.append({"type": "input_text", "text": part["text"]})
                elif part["type"] == "image_url":
                    image_url=part["image_url"]
                    image_part={"type": "input_image", "image_url": image_url["url"] if isinstance(image_url, dict) else image_url}
                    if isinstance(image_url, dict) and "detail" in image_url:
                        image_part["detail"]=image_url["detail"]
                    content_parts.append(image_part)
        openai_input.append({"role": role, "content": content_parts})
    instructions="\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, openai_input

def build_submission_dict(records:List[Dict[str,Any]])->Dict[str,Any]:
    submission={}
    for rec in sorted(records, key=lambda r: r["task_id"]):
        first=rec["pred_first"]
        second=rec["pred_second"]
        submission[rec["task_id"]]=[
            {
                "attempt_1": normalize_submission_grid(first[i]),
                "attempt_2": normalize_submission_grid(second[i]),
            }
            for i in range(rec["num_test"])
        ]
    return submission

# ---------- OpenAI call ----------
async def close_stream(stream:Any)->None:
    close_fn=getattr(stream, "close", None)
    if close_fn is None:
        return
    result=close_fn()
    if inspect.isawaitable(result):
        await result

async def call_openai_async(client:AsyncOpenAI, model:str, messages:List[Dict], temperature:float,
                            timeout_s:float, retries:int, backoff_base:float)->str:
    attempt=0
    while True:
        stream=None
        try:
            instructions, openai_input = messages_to_openai_input(messages)
            stream = await client.responses.create(
                model=normalize_model_name(model),
                instructions=instructions,
                input=openai_input,
                reasoning={"effort": "xhigh"},
                stream=True,
                timeout=timeout_s,
            )
            chunks=[]
            final_response=None
            async for event in stream:
                if event.type == "response.output_text.delta":
                    chunks.append(event.delta)
                elif event.type == "response.completed":
                    final_response = event.response
                elif event.type == "response.failed":
                    raise RuntimeError(f"OpenAI response failed: {event.response}")
                elif event.type == "error":
                    raise RuntimeError(f"OpenAI stream error: {event.error}")
            if final_response is not None and getattr(final_response, "output_text", None):
                return final_response.output_text
            text="".join(chunks).strip()
            if text:
                return text
            raise RuntimeError("OpenAI stream finished without output text.")
        except Exception as e:
            attempt+=1
            if attempt>retries: raise
            await asyncio.sleep((backoff_base**attempt)+random.uniform(0,0.25))
        finally:
            if stream is not None:
                await close_stream(stream)

# ---------- Evaluation ----------
async def eval_task_once(idx:int, total:int, sem:asyncio.Semaphore, client:AsyncOpenAI,
                         task_id:str, model:str, messages:List[Dict], golds:Optional[List[List[List[int]]]],
                         num_test:int, quiet:bool,
                         timeout_s:float, retries:int, backoff_base:float,
                         strategy:str, temp: float):
    async with sem:
        k=num_test
        try:
            reply = await call_openai_async(client, model, messages, temp, timeout_s, retries, backoff_base)
            if not quiet:
                print(reply)
            first, second = parse_multiple_grids(reply, k, log_failures=not quiet)
            base_record = {
                "idx": idx,
                "task_id": task_id,
                "strategy": strategy,
                "num_test": k,
                "pred_first": first,
                "pred_second": second,
            }
            if quiet:
                return base_record
            per_pair_exact = []
            per_pair_pcell = []
            for i in range(k):
                g= golds[i]
                a= first[i]; b= second[i]
                ok = (is_grid(a) and grids_equal(a,g)) or (is_grid(b) and grids_equal(b,g))
                per_pair_exact.append(bool(ok))
                p1 = per_cell_accuracy(a,g) if is_grid(a) else 0.0
                p2 = per_cell_accuracy(b,g) if is_grid(b) else 0.0
                per_pair_pcell.append(max(p1,p2))
            score_exact = sum(per_pair_exact)/k
            score_pcell = sum(per_pair_pcell)/k
            return {
                **base_record,
                "score_exact": score_exact,
                "score_pcell": score_pcell,
                "per_pair_exact": per_pair_exact,
                "per_pair_pcell": per_pair_pcell,
                "golds": golds,
                "raw": reply
            }
        except Exception as e:
            error_record = {
                "idx": idx,
                "task_id": task_id,
                "strategy": strategy,
                "num_test": k,
                "error": str(e),
                "pred_first": ["" for _ in range(k)],
                "pred_second": ["" for _ in range(k)],
            }
            if quiet:
                return error_record
            return {**error_record, "score_exact": 0.0, "score_pcell": 0.0, "golds": golds}

async def eval_tasks(args):
    api_key=os.environ.get("OPENAI_API_KEY")
    if not api_key: raise RuntimeError("Set OPENAI_API_KEY env var.")
    raw_tasks=load_arc_tasks(args.dataset_dir, args.limit)
    built_grid=[]
    for t in raw_tasks:
        task_data=t["task"]
        m,g=build_messages_for_task_grid(task_data, args.mode)
        built_grid.append((t["task_id"], m, g, len(task_data["test"])))
    if args.task_mode == "score":
        missing_outputs=[task_id for task_id, _, golds, _ in built_grid if golds is None]
        if missing_outputs:
            raise RuntimeError(f"Scoring mode requires test outputs. Missing outputs for tasks: {', '.join(missing_outputs[:5])}")
    sem=asyncio.Semaphore(args.concurrency)
    records=[]
    client = AsyncOpenAI(api_key=api_key, max_retries=0, timeout=args.timeout)
    try:
        tasks=[]
        total = len(raw_tasks)
        for idx,(task_id, m, g, num_test) in enumerate(built_grid, start=1):
            tasks.append(eval_task_once(
                idx, total, sem, client, task_id, args.model, m, g, num_test,
                args.task_mode == "submission", args.timeout, args.retries,
                args.backoff, "grid", args.temperature
            ))

        for coro in asyncio.as_completed(tasks):
            rec=await coro
            records.append(rec)
            if "error" in rec:
                print(f"[{rec['idx']}/{total}] ({rec['strategy']}) ERROR: {rec['error']}")
            elif args.task_mode == "score":
                print(f"[{rec['idx']}/{total}] ({rec['strategy']}) exact={rec['score_exact']:.3f}  pcell={rec['score_pcell']:.3f}")
            else:
                print(f"[{rec['idx']}/{total}] ({rec['strategy']}) completed task_id={rec['task_id']}")
    finally:
        await client.close()

    if args.task_mode == "submission":
        out_path = Path(args.out or "submission.json")
        with open(out_path, "w") as f:
            json.dump(build_submission_dict(records), f)
        print(f"\nWrote submission file to {out_path}")
        return

    def summarize(strategy_name:str):
        subset=[r for r in records if r.get("strategy")==strategy_name and "error" not in r]
        if not subset: return None
        exact= [r["score_exact"] for r in subset]
        pcell= [r["score_pcell"] for r in subset]
        return {
            "n_tasks": len(subset),
            "mean_exact": float(statistics.mean(exact)) if exact else 0.0,
            "mean_pcell": float(statistics.mean(pcell)) if pcell else 0.0
        }
    sum_grid = summarize("grid")
    print("\\n====== SUMMARY ======")
    print(f"GRID  - tasks={sum_grid['n_tasks']}  mean_exact={sum_grid['mean_exact']:.3f}  mean_pcell={sum_grid['mean_pcell']:.3f}")
    out_path = Path(args.out or f"arc_interface_results.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def main():
    parser=argparse.ArgumentParser(description="ARC eval")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task_mode", choices=["score","submission"], default="score")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.6)
    parser.add_argument("--mode", choices=["multimodal","text"], default="multimodal")
    parser.add_argument("--out", type=str, default=None, help="Path to write JSONL results")
    parser.add_argument("--temperature", type=float, default=0.0)
    args=parser.parse_args()
    random.seed(42)
    asyncio.run(eval_tasks(args))
if __name__=="__main__":
    main()
