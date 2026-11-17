import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

export const runtime = 'nodejs';

// Loss-validation results written by extensions/turbo_lora/loss_validation.py:
// <job folder>/validation/loss.jsonl, one JSON line per validation event.
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const filePath = path.join(trainingFolder, job.name, 'validation', 'loss.jsonl');

  try {
    await fs.promises.access(filePath);
  } catch {
    return NextResponse.json({ exists: false, events: [] });
  }

  // lightweight existence check used to decide if the Validation tab should show
  if (request.nextUrl.searchParams.get('check') === '1') {
    return NextResponse.json({ exists: true, events: [] });
  }

  let text = '';
  try {
    text = await fs.promises.readFile(filePath, 'utf-8');
  } catch (error) {
    console.error('Error reading validation file:', error);
    return NextResponse.json({ error: 'Error reading validation file' }, { status: 500 });
  }

  // a step recorded twice (a resumed run validates its start step again) keeps its last record;
  // a partially written last line (training is appending) is skipped
  const byStep = new Map<number, any>();
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    try {
      const event = JSON.parse(line);
      if (typeof event.step === 'number') byStep.set(event.step, event);
    } catch {
      // ignore
    }
  }
  const events = Array.from(byStep.values()).sort((a, b) => a.step - b.step);
  return NextResponse.json({ exists: true, events });
}
