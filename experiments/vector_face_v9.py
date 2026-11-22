#!/usr/bin/env python3
"""Vector face illustration on top of the EXISTING full artwork.
This is a color portrait for art approval, deliberately without numbering.
Color fills in the face are restricted to seven original paint numbers.
"""
import base64, io, json, sys
from pathlib import Path
import cairosvg
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_artistic_v4 as art

OUT=Path("face_v9_color_only")
BASE=OUT/"portrait_v9_color.png"

def main():
    if not BASE.exists():raise FileNotFoundError(BASE)
    _,palette,numbers,_,_=art.prepare()
    inv={int(num):int(key) for key,num in numbers.items()}
    hexes={num:"#"+''.join(f"{int(v):02X}" for v in palette[inv[num]])
           for num in (4,11,13,14,15,18,19)}
    for num in (4,11,13,14,15,18,19):
        if num not in inv:raise RuntimeError(f"Missing palette number {num}")
    img=Image.open(BASE).convert("RGB")
    if img.size!=(1122,1402):raise RuntimeError(f"Wrong source canvas: {img.size}")
    with io.BytesIO() as buffer:
        img.save(buffer,format="PNG")
        encoded=base64.b64encode(buffer.getvalue()).decode("ascii")
    # Base image retains the complete picture; the overlaid VECTORS
    # redraw ONLY the same local face/ear/bangs coordinates.
    f=lambda num:hexes[num]
    el=[]
    add=el.append
    add('<svg xmlns="http://www.w3.org/2000/svg" width="1122" height="1402" viewBox="0 0 1122 1402">')
    add('<defs>')
    add('<clipPath id="faceLimit"><rect x="456" y="344" width="219" height="168"/></clipPath>')
    add('<clipPath id="nearEye"><path d="M547 394 C557 383 577 380 590 390 C597 394 596 399 590 403 C573 410 555 405 547 394 Z"/></clipPath>')
    add('<clipPath id="farEye"><path d="M625 387 C634 380 647 380 657 386 C663 391 662 394 655 399 C643 402 632 397 625 387 Z"/></clipPath>')
    add('</defs>')
    add(f'<image x="0" y="0" width="1122" height="1402" href="data:image/png;base64,{encoded}"/>')
    add('<g clip-path="url(#faceLimit)" stroke-linejoin="round" stroke-linecap="round">')
    add(f'<path fill="{f(15)}" d="M520 385 C536 370 552 379 579 370 C606 358 627 358 645 377 C650 388 648 405 654 419 C661 424 662 428 654 432 C651 435 649 439 654 445 C658 449 650 449 645 451 C651 453 648 458 640 460 C645 466 637 479 623 486 C604 497 579 489 558 483 C533 473 528 455 524 436 C520 423 516 398 520 385 Z"/>')
    add(f'<path fill="{f(13)}" d="M525 450 C531 467 545 477 569 489 C585 498 613 499 631 487 C608 488 587 480 568 465 C555 458 542 452 525 450 Z"/>')
    add(f'<path fill="{f(13)}" d="M644 414 C647 420 654 424 653 430 C647 433 643 428 644 414 Z"/>')
    add(f'<path fill="{f(15)}" d="M466 405 C475 399 485 400 493 412 C502 424 498 439 483 449 C472 447 461 433 462 417 C463 409 466 405 466 405 Z"/>')
    add(f'<path fill="{f(13)}" d="M473 412 C485 407 492 416 490 426 C487 436 476 438 473 428 C469 420 470 416 473 412 Z"/>')
    # Distinct anime-eye geometry, inset irises and pupils masked by eyelid.
    add(f'<path fill="{f(19)}" d="M547 394 C557 383 577 380 590 390 C597 394 596 399 590 403 C573 410 555 405 547 394 Z"/>')
    add(f'<path fill="{f(19)}" d="M625 387 C634 380 647 380 657 386 C663 391 662 394 655 399 C643 402 632 397 625 387 Z"/>')
    add('<g clip-path="url(#nearEye)">')
    add(f'<ellipse cx="577" cy="395" rx="12" ry="10" fill="{f(14)}" transform="rotate(-8 577 395)"/>')
    add(f'<ellipse cx="582" cy="394" rx="5" ry="7" fill="{f(4)}"/>')
    add(f'<circle cx="575" cy="390" r="2.2" fill="{f(18)}"/>')
    add('</g>')
    add('<g clip-path="url(#farEye)">')
    add(f'<ellipse cx="647" cy="390" rx="9" ry="9" fill="{f(14)}"/>')
    add(f'<ellipse cx="651" cy="390" rx="4" ry="6" fill="{f(4)}"/>')
    add(f'<circle cx="645" cy="387" r="2" fill="{f(18)}"/>')
    add('</g>')
    add(f'<path d="M548 392 C561 381 579 379 594 391" stroke="{f(4)}" stroke-width="2.6" fill="none"/>')
    add(f'<path d="M549 399 C563 407 579 407 592 401" stroke="{f(4)}" stroke-width="1.35" fill="none"/>')
    add(f'<path d="M590 389 C598 387 602 384 605 378" stroke="{f(4)}" stroke-width="1.4" fill="none"/>')
    add(f'<path d="M626 385 C636 378 650 379 660 386" stroke="{f(4)}" stroke-width="2.4" fill="none"/>')
    add(f'<path d="M627 392 C640 400 650 400 659 395" stroke="{f(4)}" stroke-width="1.15" fill="none"/>')
    add(f'<path d="M651 408 C650 419 659 424 652 430" stroke="{f(4)}" stroke-width="1.2" fill="none"/>')
    add(f'<path d="M650 430 C649 432 648 432 647 430" stroke="{f(4)}" stroke-width="1.35" fill="none"/>')
    add(f'<path d="M636 452 C642 450 645 451 650 450 C646 455 643 457 639 458 C637 457 635 456 636 452" fill="{f(14)}"/>')
    add(f'<path d="M647 451 C642 451 639 453 635 453" stroke="{f(4)}" stroke-width="1.2" fill="none"/>')
    add(f'<path d="M641 457 C638 460 636 460 633 458" stroke="{f(13)}" stroke-width="1" fill="none"/>')
    add(f'<path d="M473 413 C482 407 488 410 491 419" stroke="{f(13)}" stroke-width="2.5" fill="none"/>')
    add(f'<path d="M480 428 C487 434 490 429 490 423" stroke="{f(13)}" stroke-width="2" fill="none"/>')
    add(f'<path d="M533 360 C541 375 548 378 550 389" stroke="{f(4)}" stroke-width="1.0" fill="none"/>')
    add(f'<path d="M567 350 C574 371 581 378 587 388" stroke="{f(4)}" stroke-width="1.0" fill="none"/>')
    add(f'<path d="M611 362 C620 378 625 389 629 399" stroke="{f(4)}" stroke-width="1.0" fill="none"/>')
    add('</g></svg>')
    svg='\n'.join(el)
    (OUT/"portrait_v9_smooth_face.svg").write_text(svg,encoding="utf-8")
    cairosvg.svg2png(bytestring=svg.encode("utf-8"),
       write_to=str(OUT/"portrait_v9_smooth_preview.png"),
       output_width=2244,output_height=2804)
    Image.open(OUT/"portrait_v9_smooth_preview.png").convert("RGB").crop(
        (900,688,1360,1030)).save(OUT/"smooth_face_detail.png")
    meta={"source_art":"face_v9_color_only/portrait_v9_color.png",
          "recolored_face_vector_pigments":hexes,
          "face_only_view":"vector overlay in x456..675 y344..512",
          "outside_face_source":"unchanged native indexed color map",
          "numbering_started":False}
    (OUT/"vector_preview_audit.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print("V9_VECTOR_PREVIEW",json.dumps(meta,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
