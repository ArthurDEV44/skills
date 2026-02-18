# Tokio I/O and Framing

## AsyncRead / AsyncWrite

Core traits for async byte stream I/O. Use extension traits for convenient methods:

```rust
use tokio::io::{AsyncReadExt, AsyncWriteExt};
```

### Reading

```rust
// Read into buffer, returns bytes read. Ok(0) = EOF.
let n = stream.read(&mut buf).await?;

// Read everything to end
let mut data = Vec::new();
stream.read_to_end(&mut data).await?;

// Read exactly N bytes (errors if EOF before N)
let mut exact = [0u8; 64];
stream.read_exact(&mut exact).await?;
```

### Writing

```rust
// Write buffer, may not write all bytes
let n = stream.write(b"hello").await?;

// Write ALL bytes (retries until complete)
stream.write_all(b"hello").await?;
```

### Copy

```rust
// Copy all bytes from reader to writer
tokio::io::copy(&mut reader, &mut writer).await?;

// Bidirectional copy (useful for proxies)
tokio::io::copy_bidirectional(&mut stream_a, &mut stream_b).await?;
```

### Helpers

```rust
tokio::io::stdin()   // async stdin
tokio::io::stdout()  // async stdout
tokio::io::stderr()  // async stderr
tokio::io::empty()   // yields EOF immediately
tokio::io::sink()    // discards all written data
tokio::io::repeat(0) // infinite stream of a byte
```

## Socket Splitting

Two strategies for simultaneous read/write:

### Same-task (zero-cost, reference-based)

```rust
let (mut rd, mut wr) = socket.split();
// rd and wr borrow socket, must be used in same task
tokio::io::copy(&mut rd, &mut wr).await?;
```

### Cross-task (Arc-based)

```rust
let (rd, wr) = socket.into_split();
// rd and wr own their halves via Arc, can move to different tasks
tokio::spawn(async move { read_loop(rd).await });
tokio::spawn(async move { write_loop(wr).await });
```

## Echo Server Patterns

### Using io::copy

```rust
use tokio::net::TcpListener;
use tokio::io;

let listener = TcpListener::bind("127.0.0.1:6142").await?;
loop {
    let (mut socket, _) = listener.accept().await?;
    tokio::spawn(async move {
        let (mut rd, mut wr) = socket.split();
        io::copy(&mut rd, &mut wr).await.ok();
    });
}
```

### Manual read/write loop

```rust
tokio::spawn(async move {
    let mut buf = vec![0; 1024];
    loop {
        match socket.read(&mut buf).await {
            Ok(0) => return,               // EOF
            Ok(n) => {
                socket.write_all(&buf[..n]).await.ok();
            }
            Err(_) => return,
        }
    }
});
```

## Framing

Framing converts a byte stream into discrete message frames.

### BytesMut for buffered reads

```rust
use bytes::BytesMut;

let mut buffer = BytesMut::with_capacity(4096);
```

`BytesMut` tracks read/write cursors automatically. Use `Buf` trait for reading (`get_u8()`, advancing cursor) and `BufMut` for writing.

### Read loop pattern

```rust
loop {
    // 1. Try parsing a frame from buffer
    if let Some(frame) = parse_frame(&mut buffer)? {
        return Ok(Some(frame));
    }
    // 2. Read more data from socket
    if stream.read_buf(&mut buffer).await? == 0 {
        // EOF: check for partial frame
        if buffer.is_empty() {
            return Ok(None); // clean shutdown
        } else {
            return Err("connection reset".into());
        }
    }
}
```

### Buffered writes with BufWriter

```rust
use tokio::io::BufWriter;

let stream = BufWriter::new(tcp_stream);
// Multiple small writes are batched
stream.write_u8(b'+').await?;
stream.write_all(msg.as_bytes()).await?;
stream.write_all(b"\r\n").await?;
stream.flush().await?; // ensure data reaches socket
```

### tokio_util::codec (frame codecs)

For production framing, use `tokio_util::codec`:

```rust
use tokio_util::codec::{Framed, LinesCodec, Decoder, Encoder};

let framed = Framed::new(socket, LinesCodec::new());

// Read frames
while let Some(line) = framed.next().await {
    let line = line?;
    println!("got line: {line}");
}

// Write frames
framed.send("hello".to_string()).await?;
```

Built-in codecs: `LinesCodec`, `BytesCodec`, `LengthDelimitedCodec`.

Custom codec: implement `Decoder` and `Encoder` traits.

### Tips

- Allocate buffers on the heap (`Vec`, `BytesMut`) not the stack -- keeps task struct small
- Always check for `Ok(0)` (EOF) to avoid infinite loops
- Use `bytes` crate (`Bytes`, `BytesMut`) for zero-copy buffer management
- `Bytes::clone()` is cheap (reference-counted, no data copy)
- Prefer `read_buf(&mut BytesMut)` over `read(&mut [u8])` to avoid manual cursor management

## Sources

- [I/O tutorial](https://tokio.rs/tokio/tutorial/io)
- [Framing tutorial](https://tokio.rs/tokio/tutorial/framing)
- [tokio::io module](https://docs.rs/tokio/latest/tokio/io/index.html)
- [tokio_util::codec](https://docs.rs/tokio-util/latest/tokio_util/codec/index.html)
