import { NextRequest, NextResponse } from "next/server";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";

const DATA_ROOT = path.resolve(/*turbopackIgnore: true*/ process.cwd(), "..", "synthetic_ct");

const CONTENT_TYPES: Record<string, string> = {
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp"
};

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: pathParts } = await params;
  const requested = path.resolve(DATA_ROOT, ...pathParts);
  const relative = path.relative(DATA_ROOT, requested);

  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    return NextResponse.json({ error: "Invalid asset path" }, { status: 400 });
  }

  try {
    const fileStat = await stat(requested);
    if (!fileStat.isFile()) {
      return NextResponse.json({ error: "Asset is not a file" }, { status: 404 });
    }

    const bytes = await readFile(requested);
    const contentType = CONTENT_TYPES[path.extname(requested).toLowerCase()] ?? "application/octet-stream";

    return new NextResponse(bytes, {
      headers: {
        "content-type": contentType,
        "cache-control": "public, max-age=60"
      }
    });
  } catch {
    return NextResponse.json({ error: "Asset not found" }, { status: 404 });
  }
}
