use clap::Parser;
use inline_colorization::*;
use memmap::MmapMut;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::cmp::min;
use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::string::ToString;
use std::time::{Duration, Instant};
use anyhow::Result;

const ITER_COUNT: usize = 1;
// const size: u64 = 2u64.pow(30);
// const rows: LazyCell<usize> = LazyCell::new(|| ((size as f64).sqrt().ceil() as u64).next_power_of_two() as usize);
//
// const cols: LazyCell<usize> = LazyCell::new(|| size as usize / rows);

#[derive(Parser)]
struct Cli {
    /// the log2 size of the files to test over
    #[arg(default_value_t = 20)]
    log2_size: u32,

    /// whether to show little versions of the files after each use
    #[arg(short)]
    verbose: bool,

    /// check after each algorithm that the result is correct
    #[arg(short, requires = "check")]
    check_work: bool,

    /// number of times to repeat the experiment
    #[arg(short, default_value_t = 1)]
    times: usize,

    /// run in-memory transpose
    #[arg(short, group = "check")]
    in_memory: bool,

    /// run memmap solution
    #[arg(short)]
    mmap: bool,

    /// run file cat solution
    #[arg(short)]
    join: bool,

    /// run all solutions
    #[arg(short, group = "check")]
    all: bool,
}

#[derive(Default, Debug, Clone, Copy)]
struct Dimensions {
    size: u64,
    rows: usize,
    cols: usize,
}
fn main() -> Result<()> {
    let mut cli = Cli::parse();
    let size = 2u64.pow(cli.log2_size);
    let rows = ((size as f64).sqrt().ceil() as u64).next_power_of_two() as usize;
    let cols = size as usize / rows;
    let dims = Dimensions { size, rows, cols };
    assert!(cli.times > 0, "must run a positive amount of runs");
    if cli.all {
        cli.in_memory = true;
        cli.mmap = true;
        cli.join = true;
    }

    print!("{color_blue}");
    println!(
        "running test with filesize 2**{} == {size} bytes over {ITER_COUNT} iters",
        cli.log2_size
    );
    println!("the matrix is {} by {}", cols, rows);
    print!("{color_reset}

{style_reset}

");
    assert_eq!(cols * rows, size as usize);

    print!("{color_green}");
    let target_file = PathBuf::from("input_file.md");
    let mut input_handle = setup_file(dims, &target_file)?;
    print!("{color_reset}

{style_reset}

");
    if cli.verbose {
        println!("input file looks like this:");
        sample_file(dims, &mut input_handle)?;
    }

    let mut mem_file = if cli.in_memory {
        print!("{color_magenta}");
        println!("starting in-memory transpose");

        let mut total_duration = Duration::from_secs(0);
        let mut mem_file = File::open(PathBuf::from("input_file.md"))?;
        for _ in 0..cli.times {
            let (new_mem_file, mem_dur) = in_memory(dims, &mut input_handle)?;
            total_duration += mem_dur;
            mem_file = new_mem_file;
        }
        if cli.times > 1 {
            println!(
                "{style_bold} On average it took {:?}",
                total_duration / cli.times as u32
            );
        }
        print!("{color_reset}

{style_reset}

");

        if cli.verbose {
            println!("in_memory output looks like this:");
            sample_file(dims, &mut mem_file)?;
        }
        mem_file
    } else {
        File::open(&target_file)?
    };

    if cli.mmap {
        print!("{color_yellow}");
        println!("starting memmap solution");
        let mut total_duration = Duration::from_secs(0);
        let mut mmap_file = File::open(PathBuf::from("input_file.md"))?;
        for _ in 0..cli.times {
            let (new_mmap_file, mmap_dur) = mmap_solution(dims, &target_file)?;
            total_duration += mmap_dur;
            mmap_file = new_mmap_file;
        }
        if cli.times > 1 {
            println!(
                "{style_bold} On average it took {:?}",
                total_duration / cli.times as u32
            );
        }
        print!("{color_reset}

{style_reset}

");
        if cli.verbose {
            println!("mmap file looks like this:");
            sample_file(dims, &mut mmap_file)?;
        }
        if cli.check_work && cli.in_memory {
            assert!(file_eq_assert(&mut mem_file, &mut mmap_file)?);
        }
    }

    if cli.join {
        print!("{color_cyan}");
        println!("starting tranpose with temp files");
        let mut total_duration = Duration::from_secs(0);
        let mut joined_file = File::open(PathBuf::from("input_file.md"))?;
        for _ in 0..cli.times {
            let (new_joined_file, temp_dur) = join_file_handles(dims, &mut input_handle)?;
            total_duration += temp_dur;
            joined_file = new_joined_file;
        }
        if cli.times > 1 {
            println!(
                "{style_bold} On average it took {:?}",
                total_duration / cli.times as u32
            );
        }
        print!("{color_reset}

{style_reset}

");
        if cli.verbose {
            println!("temp file looks like this:");
            sample_file(dims, &mut joined_file)?;
        }
        if cli.check_work && cli.in_memory {
            assert!(file_eq_assert(&mut mem_file, &mut joined_file)?);
        }
    }

    Ok(())
}

fn setup_file(
    Dimensions { size, .. }: Dimensions,
    target_file: &Path,
) -> Result<File> {
    let handle = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&target_file)?;
    if target_file.metadata()?.len() != size {
        println!("setting up file to work on");
        handle.set_len(size)?;
        let mut buffered_writer = BufWriter::new(&handle);
        let letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".to_string();
        let mut writeable = letters.bytes().into_iter().cycle();
        for _ in 0..size {
            buffered_writer.write(&[writeable.next().unwrap()])?;
        }
        buffered_writer.flush()?;
    }
    assert_eq!(target_file.metadata()?.len(), size);
    Ok(handle)
}

fn in_memory(
    Dimensions { size, rows, cols }: Dimensions,
    input_handle: &mut File,
) -> Result<(File, Duration)> {
    input_handle.seek(SeekFrom::Start(0))?;
    let path: PathBuf = PathBuf::from("in_memory.md");
    let mut output_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)?;

    let start_time = Instant::now();

    let mut input_buff = Vec::with_capacity(size as usize);
    let num_read_bytes = input_handle.read_to_end(&mut input_buff)?;
    assert_eq!(num_read_bytes, size as usize);
    // transpose the data in memory
    for i in 0..rows {
        for j in i..cols {
            let swap = input_buff[i * cols + j];
            input_buff[i * cols + j] = input_buff[j * cols + i];
            input_buff[j * cols + i] = swap;
        }
    }
    let duration = start_time.elapsed();
    assert_eq!(size as usize, output_file.write(&input_buff)?);

    println!("in_memory time: {:?}", duration);

    Ok((output_file, duration))
}

fn mmap_solution(
    Dimensions { rows, cols, .. }: Dimensions,
    input_path: &Path,
) -> Result<(File, Duration)> {
    let start_with_copy = Instant::now();
    let target_path: PathBuf = PathBuf::from("mmap.md");
    std::fs::copy(input_path, &target_path)?;
    let output_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&target_path)?;
    // output_file.set_len(size)?;
    let mut mmap = unsafe { MmapMut::map_mut(&output_file)? };

    let start_time = Instant::now();
    for i in 0..rows {
        for j in i..cols {
            let swap = mmap[i * cols + j];
            mmap[i * cols + j] = mmap[j * cols + i];
            mmap[j * cols + i] = swap;
        }
    }

    let duration = start_time.elapsed();
    let duration_with_copy = start_with_copy.elapsed();
    println!("memmap time: {:?}", duration);
    println!("with the initial copy: {:?}", duration_with_copy);

    Ok((output_file, duration_with_copy))
}

fn join_file_handles(
    Dimensions { size, rows, cols }: Dimensions,
    input_handle: &mut File,
) -> Result<(File, Duration)> {
    input_handle.seek(SeekFrom::Start(0))?;
    let temp_dir = std::env::temp_dir().join("transpose_columns");
    create_dir_all(&temp_dir)?;
    let target_output = PathBuf::from("catted_cols.md");
    let mut output_file = OpenOptions::new()
        .write(true)
        .read(true)
        .create(true)
        .truncate(true)
        .open(&target_output)?;
    output_file.set_len(size)?;

    let start_time_with_temps = Instant::now();
    let mut file_io_result = || -> Result<_> {
        let mut new_row_file_handles = (0..rows)
            .map(|i| {
                let temp_file_name = temp_dir.join(format!("row-{}.md", i));
                let temp_file_handle = OpenOptions::new()
                    .write(true)
                    .read(true)
                    .create(true)
                    .truncate(true)
                    .open(&temp_file_name)?;
                let temp_file_buff_writer = BufWriter::new(temp_file_handle);
                Ok((temp_file_name, temp_file_buff_writer))
            })
            .collect::<Result<Vec<(PathBuf, BufWriter<File>)>>>();

        new_row_file_handles
    };

    let mut new_row_file_handles = file_io_result().or_else(
        |e: anyhow::Error| -> Result<_> {
            (0..rows).for_each(|i| {
                let temp_file_name = temp_dir.join(format!("row-{}.md", i));
                if temp_file_name.exists() {
                    std::fs::remove_file(&temp_file_name).unwrap();
                }
            });
            panic!("{}", e)
        },
    )?;

    let start_time = Instant::now();
        let mut row_buf = vec![0u8; cols];
        for _ in 0..rows {
            input_handle.read(&mut row_buf)?;
            (&mut row_buf, &mut new_row_file_handles)
                .into_par_iter()
                .for_each(|(input_byte, output_row)| {
                    output_row.1.write(&[*input_byte]).unwrap();
                })
        }

        let remainder_handles = new_row_file_handles
            .into_iter()
            .map(|(handle, mut writer)| {
                writer.flush().unwrap();
                std::io::copy(&mut File::open(&handle).unwrap(), &mut output_file).unwrap();
                handle
            })
            .collect::<Vec<_>>();
        output_file.flush()?;
        let duration = start_time.elapsed();

        remainder_handles
            .into_iter()
            .for_each(|handle| std::fs::remove_file(handle).unwrap());

        let duration_with_temp = start_time_with_temps.elapsed();

    println!(
        "tranpose with temp files time: {:?}\ntime with all cleanup {:?}",
        duration, duration_with_temp
    );

    Ok((output_file, duration))
}

fn sample_file(Dimensions { cols, .. }: Dimensions, file: &mut File) -> Result<()> {
    file.seek(SeekFrom::Start(0))?;
    let read_in_bytes = min(8usize, cols);
    let mut input_buf = vec![0u8; read_in_bytes];
    for _ in 0..read_in_bytes {
        file.read(input_buf.as_mut_slice())?;
        println!("{}", String::from_utf8_lossy(input_buf.as_slice()));
        file.seek_relative(cols as i64 - input_buf.len() as i64)?
    }
    Ok(())
}

fn file_eq_assert(file_a: &mut File, file_b: &mut File) -> Result<bool> {
    if file_a.metadata()?.len() != file_b.metadata()?.len() {
        return Ok(false);
    }

    file_a.seek(SeekFrom::Start(0))?;
    file_b.seek(SeekFrom::Start(0))?;

    let input_size = 2usize.pow(10);
    let mut input_buf_a = Vec::with_capacity(input_size);
    let mut input_buf_b = Vec::with_capacity(input_size);

    while file_a.read(&mut input_buf_a)? > 0 {
        file_b.read(&mut input_buf_b)?;
        if input_buf_a != input_buf_b {
            return Ok(false);
        }
    }

    Ok(true)
}

#[unsafe(no_mangle)]
fn calculate_index(i: usize, len: usize) -> usize {
    (i * 257) % len
}
