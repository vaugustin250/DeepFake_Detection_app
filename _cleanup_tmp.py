with open('main (4).py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_del = None
end_del   = None
for i, line in enumerate(lines):
    if 'def _run_trained_model' in line and i > 2400:
        start_del = i
    if start_del and i > start_del + 2 and line.startswith('    def '):
        end_del = i
        break
    # Also stop at class boundary
    if start_del and i > start_del + 2 and line.startswith('class '):
        end_del = i
        break

# If no next method found, stop at the DashboardTab comment
if start_del and not end_del:
    for i, line in enumerate(lines):
        if i > start_del and '# TAB: DASHBOARD' in line:
            end_del = i
            break

print(f'start_del={start_del}, end_del={end_del}')
if start_del and end_del:
    new_lines = lines[:start_del] + lines[end_del:]
    with open('main (4).py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f'Deleted {end_del-start_del} lines. New length: {len(new_lines)}')
else:
    print('Could not find range')
    for i, line in enumerate(lines):
        if i >= (start_del or 2639) and i < (start_del or 2639) + 10:
            print(i+1, repr(line[:80]))
