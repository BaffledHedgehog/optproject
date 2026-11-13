"""Create local v20 PDFs and self-contained source/evidence archives. No upload."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write_zip(path, members):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, (2026,11,14,12,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info,data)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(members)
        for name,data in members.items():
            assert sha(archive.read(name)) == sha(data), name

def main():
    report = json.loads((ROOT/'Evidence/checks/release_validation.json').read_text())
    assert report['replayed_runs'] == 432
    for name,digest in report['source_sha256'].items():
        assert sha((ROOT/name).read_bytes()) == digest, 'Re-run validation after changing '+name
    for lang in ('en','ru'):
        folder = ROOT/'Paper' / ('ru' if lang=='ru' else '')
        receipt = json.loads((folder/'build_receipt.json').read_text())
        assert len(receipt['passes'])==3 and all(p['exit']==0 for p in receipt['passes'])
        assert not receipt['warning_lines']
        for name,digest in receipt['source_hashes'].items():
            assert sha((folder/name).read_bytes())==digest, 'Rebuild PDF after changing '+name
        pdf = folder/('paper_ru.pdf' if lang=='ru' else 'paper.pdf')
        final_pdf = ROOT/f'paper_{lang}_v20.pdf'
        if pdf.exists():
            shutil.copyfile(pdf, final_pdf)
        else:
            previous = json.loads((ROOT/'SHA256SUMS.json').read_text())
            assert sha(final_pdf.read_bytes()) == previous[final_pdf.name], 'Rebuild changed PDF'
        members = {p.name:p.read_bytes().replace(b'../figures/',b'figures/') for p in folder.glob('*.tex')}
        members.update({'figures/'+p.name:p.read_bytes() for p in (ROOT/'Paper/figures').glob('*.pdf')
                        if p.stem.endswith('_'+lang)})
        members['00README.txt'] = ('Compile article.tex three times with '+('xelatex' if lang=='ru' else 'pdflatex')+
            '. No scientific rerun is needed. See the full reading packet for code and evidence.\n').encode()
        name = 'arxiv_source_ru_v20.zip' if lang=='ru' else 'arxiv_source_v20.zip'
        write_zip(ROOT/name,members)
    members = {}
    for folder in ('Code','Evidence','Paper'):
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or any(part in ('__pycache__','smoke','layout') or part.startswith('reproduced_') for part in p.parts):
                continue
            if folder=='Paper':
                # Keep sources, figures and receipts; final PDFs live at the root.
                if p.suffix not in ('.tex','.pdf','.png','.json') or p.name in ('paper.pdf','paper_ru.pdf'):
                    continue
            members[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    for p in ROOT.iterdir():
        if p.is_file() and (p.suffix in ('.py','.txt') or p.name in (
                'paper_en_v20.pdf','paper_ru_v20.pdf','arxiv_source_v20.zip','arxiv_source_ru_v20.zip')):
            members[p.name] = p.read_bytes()
    manifest = {name:sha(data) for name,data in sorted(members.items())}
    manifest_bytes = (json.dumps(manifest,indent=2)+'\n').encode()
    (ROOT/'SHA256SUMS.json').write_bytes(manifest_bytes)
    members['SHA256SUMS.json'] = manifest_bytes
    packet = ROOT/'reading_packet_v20.zip'
    write_zip(packet,members)
    (ROOT/'release_artifacts.json').write_text(json.dumps({
        name: {'bytes':(ROOT/name).stat().st_size,'sha256':sha((ROOT/name).read_bytes())}
        for name in ('paper_en_v20.pdf','paper_ru_v20.pdf','arxiv_source_v20.zip','arxiv_source_ru_v20.zip','reading_packet_v20.zip')},indent=2)+'\n')
    print('Verified',len(members),'members;',packet.name,packet.stat().st_size,'bytes')

if __name__=='__main__':
    main()
