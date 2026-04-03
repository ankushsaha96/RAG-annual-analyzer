# Quick Start - Fixed Version

## 🚀 Quick Setup (5 minutes)

### Step 1: Verify Setup
```bash
cd /Users/ankushsaha/Desktop/RAG\ -\ Annual\ result\ analyzer

# Check Python and packages
python test_simple.py
```

Expected output: `✓ All tests passed!`

### Step 2: Set API Key (if not already set)
```bash
export GROQ_API_KEY="your-api-key-from-groq-console"
```

Verify it's set:
```bash
echo $GROQ_API_KEY
```

### Step 3: Create Embeddings (One-time, ~2-3 minutes)
```bash
python main.py embed
```

This creates `Data/embedding.csv` which you'll reuse.

### Step 4: Start Querying!

**Option A: Single Query**
```bash
python main.py query "What is TCS's strategy?"
```

**Option B: Interactive Mode**
```bash
python main.py interactive
```

Type multiple questions, press Enter after each.
Type `exit` or `quit` to exit.

**Option C: Safe Interactive (for troubleshooting)**
```bash
python safe_interactive.py
```

Provides step-by-step guidance.

## ✅ Verification Checklist

- [ ] `python test_simple.py` runs without segfault
- [ ] `echo $GROQ_API_KEY` shows your API key
- [ ] `ls -lh Data/embedding.csv` shows the embeddings file exists (after running `embed`)
- [ ] `python main.py query "test"` runs successfully
- [ ] `python main.py interactive` works without crashing

## 🔧 Troubleshooting

### Issue: Still getting segmentation fault?

**Try these in order:**

1. Restart terminal and check API key:
   ```bash
   # Close and reopen terminal
   export GROQ_API_KEY="your-api-key"
   echo $GROQ_API_KEY
   ```

2. Test components separately:
   ```bash
   python test_simple.py          # Test basic imports
   python main.py embed           # Create embeddings
   ```

3. Use safe mode:
   ```bash
   python safe_interactive.py     # Guided mode
   ```

4. Check memory:
   ```bash
   # macOS: see available memory
   vm_stat | grep "Pages free"
   ```
   Need at least 2-4 GB free.

### Issue: "GROQ_API_KEY not provided"?

```bash
# Set it temporarily
export GROQ_API_KEY="your-key"

# Or permanently (add to ~/.zshrc or ~/.bash_profile)
echo 'export GROQ_API_KEY="your-key"' >> ~/.zshrc
source ~/.zshrc
```

### Issue: "Embeddings not found"?

```bash
# Create them first
python main.py embed

# Should create Data/embedding.csv
ls -lh Data/embedding.csv
```

### Issue: Everything is slow?

The first query takes 7-10 seconds. Subsequent queries are faster (3-5 sec).

This is normal because it:
1. Loads the embedding model (~3 sec)
2. Searches FAISS index (~1-2 sec)
3. Calls Groq LLM (~3-5 sec)

## 📊 Expected Performance

| Operation | Time |
|-----------|------|
| First import | ~15-30 sec (model downloads) |
| Creating embeddings | ~2-3 min (CPU) |
| Single query | 7-10 sec |
| Subsequent queries | 3-5 sec (model cached) |
| Interactive startup | ~30 sec (model loads once) |

## 🎯 What Each Command Does

```bash
# Create embeddings from PDF (one-time)
python main.py embed

# Ask a single question
python main.py query "Your question?"

# Ask multiple questions interactively
python main.py interactive

# Verbose mode (debug)
python main.py query "test" -v

# Get help
python main.py --help
```

## 📁 Important Files

- `Data/embedding.csv` - Generated embeddings (created on first `embed`)
- `main.py` - CLI application (updated with safety fixes)
- `src/config.py` - Configuration
- `src/embedding.py` - Embedding generation (updated with safety fixes)
- `src/rag.py` - RAG pipeline (updated with safety fixes)

## 🆘 Getting Help

1. Read `SEGFAULT_FIX.md` for detailed issue analysis
2. Check `README.md` for general usage
3. Check `DEVELOPMENT.md` for technical details
4. Run with `-v` flag for verbose output: `python main.py query "q" -v`

## 🎓 Learning Path

1. **First run:** `python main.py embed && python main.py interactive`
2. **Explore:** Try different questions
3. **Extend:** Read `DEVELOPMENT.md` to customize
4. **Deploy:** Read `DEPLOYMENT.md` for production

## ⚡ Pro Tips

- Create embeddings once, reuse many times
- Use `interactive` mode for exploring
- Use `query` command for scripting
- Check logs with `-v` flag if issues

You're all set! 🚀

Try: `python main.py embed && python main.py query "What is TCS's annual revenue?"`
