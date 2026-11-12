"""Manual Telegram wizard for beam analysis. No LLM or external AI API is used."""
import asyncio, logging, os, time, math, re
from collections import defaultdict, deque
from typing import Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from beam_analyser import examples, build_model
from solver_service import analyse, BeamInputError
from ai_solver import solve_direct, DirectSolverError

TOKEN=os.getenv("TELEGRAM_TOKEN_ID")
LOG_LEVEL=os.getenv("LOG_LEVEL","INFO").upper()
logging.basicConfig(level=getattr(logging,LOG_LEVEL,logging.INFO),format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger=logging.getLogger("beam_bot")
wizards: dict[int,dict[str,Any]]={}
pending: dict[int,dict[str,Any]]={}
direct_modes: dict[int,int]={}
request_times: dict[int,deque[float]]=defaultdict(deque)

def allow_request(uid):
    now=time.monotonic(); q=request_times[uid]
    while q and now-q[0]>60: q.popleft()
    if len(q)>=30: return False
    q.append(now); return True

def num(raw,label):
    try: v=float(raw.strip().replace(",",""))
    except ValueError: raise ValueError(f"Enter a valid number for {label}.")
    if not math.isfinite(v): raise ValueError(f"{label} must be finite.")
    return v

def count(raw,label,maximum=20):
    try: v=int(raw.strip())
    except ValueError: raise ValueError(f"Enter a whole number for {label}.")
    if not 0<=v<=maximum: raise ValueError(f"{label} must be between 0 and {maximum}.")
    return v

def prompt(w):
    # Build only the active prompt. A dictionary of f-strings evaluates every
    # entry eagerly and was crashing /new before the support counts existed.
    s = w["stage"]
    i = w.get("index", 0) + 1
    if s == "length":
        return "STEP 1 — Enter total beam length L in metres. Example: 4"
    if s == "stiffness_mode":
        return "STEP 2 — How is stiffness given? Reply EI (direct flexural rigidity) or E (Young's modulus plus I)."
    if s == "ei_value":
        return "Enter EI with unit: 65 MNm2, 65000 kNm2, or 65000000 Nm2."
    if s == "e_value":
        return "Enter Young's modulus E in GPa. Example: 200"
    if s == "i_value":
        return "Enter second moment of area I in m^4. Example: 0.000083"
    if s == "support_count":
        return "How many supports? Enter 1–8. Cantilever: 1; simply supported: 2; continuous beam: 3 or more."
    if s in {"support_type", "support_x"}:
        total = w.get("counts", {}).get("supports", "?")
        if s == "support_type":
            return f"Support {i}/{total}: enter fixed or pin (roller is treated as pin)."
        return f"Support {i}/{total}: enter its x-position in metres from the LEFT end."
    if s == "point_count":
        return "How many concentrated point loads? Enter 0–20."
    if s in {"point_p", "point_x"}:
        total = w.get("counts", {}).get("points", "?")
        if s == "point_p":
            return f"Point load {i}/{total}: enter magnitude in kN. Downward positive; upward negative."
        return f"Point load {i}/{total}: enter position in metres from the LEFT end."
    if s == "dist_count":
        return "How many distributed loads? Enter 0–20. Each UDL or UVL is one load."
    if s in {"dist_start", "dist_end", "dist_w1", "dist_w2"}:
        total = w.get("counts", {}).get("distributed", "?")
        if s == "dist_start":
            return f"Distributed load {i}/{total}: enter start position in metres."
        if s == "dist_end":
            return f"Distributed load {i}: enter end position in metres."
        if s == "dist_w1":
            return f"Distributed load {i}: enter intensity at START in kN/m (downward positive)."
        return f"Distributed load {i}: enter intensity at END in kN/m. For UDL, repeat the start intensity."
    if s == "moment_count":
        return "How many applied couples/moments? Enter 0–20."
    if s in {"moment_m", "moment_x"}:
        total = w.get("counts", {}).get("moments", "?")
        if s == "moment_m":
            return f"Couple {i}/{total}: enter magnitude in kN·m. Clockwise positive; anticlockwise negative."
        return f"Couple {i}: enter position in metres from the LEFT end (x=0 is unsupported)."
    if s == "query_points":
        return "Enter positions for slope/deflection values, comma-separated in metres (e.g. 1,2,3.5,4), or enter none."
    return "Unknown input step. Send /cancel and then /new."

async def ask(msg,w): await msg.reply_text(prompt(w))

def build_config(w):
    cfg={"L":w["L"],"section":w["section"],"supports":w["supports"],"point_loads":w["point_loads"],"distributed_loads":w["distributed_loads"],"moments":w["moments"],"query_points":w["query_points"]}
    build_model(cfg)
    return cfg

def summary(c):
    s=c["section"]
    stiff=(f"EI = {next(v for k,v in s.items() if k.startswith('EI_'))} ({next(k for k in s if k.startswith('EI_'))})"
           if s["type"]=="EI" else f"E = {c['E_GPa']} GPa; I = {s['I_m4']} m^4")
    lines=["BEAM INPUT SUMMARY",f"Length: {c['L']} m",f"Stiffness: {stiff}","Supports:"]
    lines += [f"• {x['type']} at x={x['x']} m" for x in c["supports"]]
    lines += ["Point loads:"] + ([f"• {x['P_kN']} kN at x={x['a']} m" for x in c["point_loads"]] or ["• None"])
    lines += ["Distributed loads:"] + ([f"• {x['w1_kNm']} → {x['w2_kNm']} kN/m from {x['start']} to {x['end']} m" for x in c["distributed_loads"]] or ["• None"])
    lines += ["Applied couples:"] + ([f"• {x['M_kNm']} kN·m at x={x['a']} m" for x in c["moments"]] or ["• None"])
    lines += ["Query positions: "+(", ".join(map(str,c["query_points"]))+" m" if c["query_points"] else "standard solver stations"),"","Confirm only after checking every value."]
    return "\n".join(lines)

async def reply_long(msg,text):
    while len(text)>3900:
        cut=text.rfind("\n",0,3900)
        if cut<1800: cut=3900
        await msg.reply_text(text[:cut].strip()); text=text[cut:].strip()
    if text: await msg.reply_text(text)

async def solve_reply(chat_id,msg,context,cfg):
    try:
        await msg.edit_text("Validating and solving with the numerical beam solver...")
        result=await asyncio.wait_for(asyncio.to_thread(analyse,cfg),timeout=30)
        warning="" if result["equilibrium_ok"] else "\nWARNING: equilibrium check failed; treat this result as unverified."
        await msg.edit_text("Calculation complete. Sending text report...")
        await reply_long(msg,result["report"]+warning)
    except asyncio.TimeoutError: await msg.edit_text("Calculation exceeded 30 seconds. Check the input.")
    except BeamInputError as e: await msg.edit_text(f"Solver validation failed: {e}")
    except Exception:
        logger.exception("Solver failure for chat %s",chat_id); await msg.edit_text("Calculation failed unexpectedly. Check Railway logs.")

async def start(update,context):
    await update.message.reply_text("BEAM ANALYSIS ASSISTANT — MANUAL INPUT MODE\n\nI ask for the values step by step, validate them, show a summary and run the numerical solver.\n\n/new starts input • /example runs a built-in test • /help shows supported values • /cancel stops")

async def help_cmd(update,context):
    await update.message.reply_text("SUPPORTED INPUTS\n\nBeam layouts: cantilever, simply supported, overhanging, propped cantilever and continuous beams, if represented by fixed/pin supports.\n\nStiffness: EI directly (kN·m², MN·m² or N·m²), or E in GPa plus I in m⁴.\nSupports: fixed or pin/roller with positions from the left end.\nLoads: multiple point loads, UDLs, UVLs (start/end intensities), and applied couples.\nSign convention: downward loads positive, upward negative, clockwise couples positive.\n\nConstant EI only. Not supported: springs, settlement, variable EI, frames, axial/inclined loads or shear deformation.\n\nEnter each alternative case (a), (b), etc. as a separate /new session.")

async def new_cmd(update,context):
    cid=update.effective_chat.id
    pending.pop(cid,None); wizards.pop(cid,None); direct_modes.pop(cid,None)
    keyboard=InlineKeyboardMarkup([
        [InlineKeyboardButton("1. Enter values manually",callback_data="mode:manual")],
        [InlineKeyboardButton("2. Direct Solution",callback_data="mode:direct")]
    ])
    await update.message.reply_text("Choose how you want to solve this cantilever beam problem:",reply_markup=keyboard)

async def mode_callback(update,context):
    q=update.callback_query
    await q.answer()
    cid=q.message.chat_id
    if q.data=="mode:manual":
        direct_modes.pop(cid,None)
        w={"stage":"length","index":0,"counts":{},"supports":[],"point_loads":[],"distributed_loads":[],"moments":[],"query_points":[],"user_id":q.from_user.id}
        wizards[cid]=w
        await q.edit_message_text("Manual numerical mode selected. Enter the requested values step by step.")
        await ask(q.message,w)
    elif q.data=="mode:direct":
        wizards.pop(cid,None); pending.pop(cid,None)
        direct_modes[cid]=q.from_user.id
        await q.edit_message_text("DIRECT SOLUTION MODE\n\nSend the complete cantilever beam question as text, or upload a clear photo/screenshot. The system will interpret the data and produce a worked solution. This mode does not call the Python beam solver.\n\nSend /cancel to exit.")

async def run_direct(update,context,text_input="",image_bytes=None,mime_type=None):
    msg=update.effective_message
    status=await msg.reply_text("Analysing your beam problem. This may take a little time...")
    try:
        answer=await asyncio.wait_for(asyncio.to_thread(solve_direct,text_input,image_bytes,mime_type),timeout=150)
        await status.edit_text("Direct solution completed. Sending worked solution...")
        await reply_long(msg,answer)
    except asyncio.TimeoutError:
        await status.edit_text("The request timed out. Please retry with a shorter prompt or clearer image.")
    except (DirectSolverError,ValueError) as exc:
        await status.edit_text(f"Direct solution could not be completed: {exc}")
    except Exception:
        logger.exception("Direct solution failed for chat %s",update.effective_chat.id)
        await status.edit_text("Direct solution failed unexpectedly. Check Railway logs.")

async def image_handler(update,context):
    cid=update.effective_chat.id
    uid=update.effective_user.id
    if direct_modes.get(cid)!=uid:
        await update.effective_message.reply_text("To solve an image, send /new and choose Direct Solution.")
        return
    msg=update.effective_message
    try:
        if msg.photo:
            tgfile=await context.bot.get_file(msg.photo[-1].file_id)
            data=bytes(await tgfile.download_as_bytearray())
            await run_direct(update,context,msg.caption or "",data,"image/jpeg")
        elif msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/"):
            tgfile=await context.bot.get_file(msg.document.file_id)
            data=bytes(await tgfile.download_as_bytearray())
            await run_direct(update,context,msg.caption or "",data,msg.document.mime_type)
        else:
            await msg.reply_text("Please upload a supported image.")
    except Exception:
        logger.exception("Image download/analysis failed for chat %s",cid)
        await msg.reply_text("Could not read that image. Please retry with a JPEG, PNG or WebP image.")

async def cancel(update,context):
    cid=update.effective_chat.id
    wizards.pop(cid,None); pending.pop(cid,None); direct_modes.pop(cid,None)
    await update.message.reply_text("Current session cancelled. Send /new to choose a mode.")

async def example_cmd(update,context):
    name=(context.args[0] if context.args else "5.2").lower(); available=examples()
    if name not in available:
        await update.message.reply_text("Available examples: "+", ".join(available)); return
    msg=await update.message.reply_text(f"Running built-in example {name}...")
    await solve_reply(update.effective_chat.id,msg,context,available[name])

async def wizard_text(update,context):
    cid=update.effective_chat.id; uid=update.effective_user.id; msg=update.effective_message
    raw=(msg.text or "").strip()
    if direct_modes.get(cid)==uid:
        await run_direct(update,context,raw)
        return
    w=wizards.get(cid)
    if not w: await msg.reply_text("Send /new and choose a solving mode."); return
    if w["user_id"]!=uid: await msg.reply_text("Only the user who started this input can continue."); return
    if not allow_request(uid): await msg.reply_text("Please slow down and send the next value in a moment."); return
    try:
        s=w["stage"]
        if s=="length":
            w["L"]=num(raw,"beam length")
            if w["L"]<=0: raise ValueError("Beam length must be positive.")
            w["stage"]="stiffness_mode"
        elif s=="stiffness_mode":
            mode=raw.lower()
            if mode not in {"ei","e"}: raise ValueError("Reply EI or E.")
            w["stage"]="ei_value" if mode=="ei" else "e_value"
        elif s=="ei_value":
            m=re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*(mnm2|mn\.m2|knm2|kn\.m2|nm2|n\.m2)\s*",raw.lower().replace(" ",""))
            if not m: raise ValueError("Enter a value and unit, e.g. 65 MNm2.")
            v=num(m.group(1),"EI")
            if v<=0: raise ValueError("EI must be positive.")
            u=m.group(2); key="EI_MNm2" if u.startswith("mn") else "EI_kNm2" if u.startswith("kn") else "EI_Nm2"
            w["section"]={"type":"EI",key:v}; w["stage"]="support_count"
        elif s=="e_value":
            w["E_GPa"]=num(raw,"E")
            if w["E_GPa"]<=0: raise ValueError("E must be positive.")
            w["stage"]="i_value"
        elif s=="i_value":
            v=num(raw,"I")
            if v<=0: raise ValueError("I must be positive.")
            w["section"]={"type":"custom","I_m4":v}; w["stage"]="support_count"
        elif s=="support_count":
            n=count(raw,"supports",8)
            if n<1: raise ValueError("At least one support is required.")
            w["counts"]["supports"]=n; w["index"]=0; w["stage"]="support_type"
        elif s=="support_type":
            kind=raw.lower()
            if kind not in {"fixed","pin","roller"}: raise ValueError("Enter fixed, pin or roller.")
            w["current_support"]="pin" if kind=="roller" else kind; w["stage"]="support_x"
        elif s=="support_x":
            x=num(raw,"support position")
            if not 0<=x<=w["L"]: raise ValueError(f"Position must be between 0 and {w['L']} m.")
            w["supports"].append({"type":w.pop("current_support"),"x":x}); w["index"]+=1
            w["stage"]="support_type" if w["index"]<w["counts"]["supports"] else "point_count"
        elif s=="point_count":
            n=count(raw,"point loads"); w["counts"]["points"]=n; w["index"]=0; w["stage"]="point_p" if n else "dist_count"
        elif s=="point_p":
            w["current_P"]=num(raw,"point load"); w["stage"]="point_x"
        elif s=="point_x":
            x=num(raw,"point-load position")
            if not 0<=x<=w["L"]: raise ValueError("Point-load position is outside the beam.")
            w["point_loads"].append({"P_kN":w.pop("current_P"),"a":x}); w["index"]+=1
            w["stage"]="point_p" if w["index"]<w["counts"]["points"] else "dist_count"
        elif s=="dist_count":
            n=count(raw,"distributed loads"); w["counts"]["distributed"]=n; w["index"]=0; w["stage"]="dist_start" if n else "moment_count"
        elif s=="dist_start":
            w["current_start"]=num(raw,"load start"); w["stage"]="dist_end"
        elif s=="dist_end":
            x=num(raw,"load end")
            if not 0<=w["current_start"]<x<=w["L"]: raise ValueError("Need 0 ≤ start < end ≤ beam length.")
            w["current_end"]=x; w["stage"]="dist_w1"
        elif s=="dist_w1":
            w["current_w1"]=num(raw,"start intensity"); w["stage"]="dist_w2"
        elif s=="dist_w2":
            w["distributed_loads"].append({"start":w.pop("current_start"),"end":w.pop("current_end"),"w1_kNm":w.pop("current_w1"),"w2_kNm":num(raw,"end intensity")})
            w["index"]+=1; w["stage"]="dist_start" if w["index"]<w["counts"]["distributed"] else "moment_count"
        elif s=="moment_count":
            n=count(raw,"couples"); w["counts"]["moments"]=n; w["index"]=0; w["stage"]="moment_m" if n else "query_points"
        elif s=="moment_m":
            w["current_M"]=num(raw,"couple"); w["stage"]="moment_x"
        elif s=="moment_x":
            x=num(raw,"couple position")
            if not 0<x<=w["L"]: raise ValueError("Couple position must satisfy 0 < x ≤ L.")
            w["moments"].append({"M_kNm":w.pop("current_M"),"a":x}); w["index"]+=1
            w["stage"]="moment_m" if w["index"]<w["counts"]["moments"] else "query_points"
        elif s=="query_points":
            if raw.lower() not in {"none","no","skip"}:
                xs=[num(v,"query position") for v in raw.split(",")]
                if any(x<0 or x>w["L"] for x in xs): raise ValueError("Query positions must be within the beam.")
                w["query_points"]=xs
            cfg=build_config(w); pending[cid]={"user_id":uid,"config":cfg}; wizards.pop(cid,None)
            kb=InlineKeyboardMarkup([[InlineKeyboardButton("Confirm & solve",callback_data="manual:confirm"),InlineKeyboardButton("Cancel",callback_data="manual:cancel")]])
            await msg.reply_text(summary(cfg),reply_markup=kb); return
        await ask(msg,w)
    except ValueError as e:
        await msg.reply_text(str(e)+"\n\n"+prompt(w))

async def callback(update,context):
    q=update.callback_query; await q.answer(); cid=q.message.chat_id; state=pending.get(cid)
    if not state or state["user_id"]!=q.from_user.id:
        await q.edit_message_text("Confirmation expired. Send /new."); return
    action=q.data.split(":")[1]
    pending.pop(cid,None)
    if action=="cancel": await q.edit_message_text("Cancelled. Send /new to start again."); return
    await solve_reply(cid,q.message,context,state["config"])

async def error_handler(update,context): logger.error("Unhandled Telegram handler error",exc_info=context.error)

def main():
    if not TOKEN: raise RuntimeError("TELEGRAM_TOKEN_ID environment variable is missing.")
    app=Application.builder().token(TOKEN).concurrent_updates(1).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CommandHandler("help",help_cmd))
    app.add_handler(CommandHandler("new",new_cmd))
    app.add_handler(CommandHandler("cancel",cancel))
    app.add_handler(CommandHandler("example",example_cmd))
    app.add_handler(CallbackQueryHandler(mode_callback,pattern=r"^mode:(manual|direct)$"))
    app.add_handler(CallbackQueryHandler(callback,pattern=r"^manual:(confirm|cancel)$"))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE,image_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,wizard_text))
    app.add_error_handler(error_handler)
    logger.info("Starting beam bot with independent manual solver and optional direct-solution mode.")
    app.run_polling(drop_pending_updates=False,allowed_updates=Update.ALL_TYPES)

if __name__=="__main__": main()
