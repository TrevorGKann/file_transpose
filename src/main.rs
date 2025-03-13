use anyhow::Result;
use clap::Parser;
use inline_colorization::*;
use memmap::{Mmap, MmapMut};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use size::Size;
use std::cmp::min;
use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::prelude::FileExt;
use std::path::{Path, PathBuf};
use std::string::ToString;
use std::time::{Duration, Instant};

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
    #[arg(short)]
    check_work: bool,

    /// number of times to repeat the experiment
    #[arg(short, default_value_t = 1)]
    times: usize,

    /// run in-memory transpose
    #[arg(short)]
    in_memory: bool,

    /// run memmap solution
    #[arg(short)]
    mmap: bool,

    /// run file cat solution
    #[arg(short)]
    join: bool,

    /// run the transpose entirely on disk
    #[arg(short)]
    on_disk: bool,

    /// entirely on disk but with a write_buffer to minimize the writes
    #[arg(short)]
    buff_on_disk: bool,

    /// **toggle** all solutions.
    /// Doing -a and -b, for example, will run all solutions except the
    /// buffered one.
    #[arg(short)]
    all: bool,
}

#[derive(Default, Debug, Clone, Copy)]
struct Dimensions {
    size: u64,
    rows: usize,
    cols: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    _main(cli)
}

// for mockup tests
fn _main(mut cli: Cli) -> Result<()> {
    let size = 2u64.pow(cli.log2_size);
    let rows = ((size as f64).sqrt().ceil() as u64).next_power_of_two() as usize;
    let cols = size as usize / rows;
    assert!(rows >= cols, "for convenience, wlog, rows >= cols");
    let dims = Dimensions { size, rows, cols };
    assert!(cli.times > 0, "must run a positive amount of runs");
    if cli.all {
        cli.in_memory ^= true;
        cli.mmap ^= true;
        cli.on_disk ^= true;
        cli.buff_on_disk ^= true;
        cli.join ^= true;
    }
    assert!(
        !cli.check_work || cli.in_memory,
        "{color_red}the in_memory solution is used as the reference solution, and therefore must be on to check work.{color_reset}"
    );

    // setup file
    print!("{color_blue}");
    println!(
        "running test with filesize 2**{} == {} over {ITER_COUNT} iters",
        cli.log2_size,
        Size::from_bytes(size),
    );
    println!("the matrix is {cols}cols by {rows}rows");
    print!("{color_reset}{style_reset}\n");
    assert_eq!(cols * rows, size as usize);

    print!("{color_green}");
    let target_file = PathBuf::from("input_file.md");
    let mut input_handle = setup_file(dims, &target_file)?;
    print!("{color_reset}{style_reset}\n");
    if cli.verbose {
        println!("input file looks like this:");
        sample_file(dims, &mut input_handle)?;
    }

    // in memory solution
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
                "{style_bold}On average it took {:?}",
                total_duration / cli.times as u32
            );
        }
        print_throughput(size * cli.times as u64, total_duration);
        print!("{color_reset}{style_reset}\n");

        if cli.verbose {
            println!("in_memory output looks like this:");
            sample_file(dims, &mut mem_file)?;
        }
        if cli.check_work {
            mem_file.seek(SeekFrom::Start(0))?;
            let mut temp_storage = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(PathBuf::from("temp_transpose_file.md"))?;
            std::io::copy(&mut mem_file, &mut temp_storage)?;
            temp_storage.seek(SeekFrom::Start(0))?;
            let (mut new_mem_file, _) = in_memory(dims, &mut temp_storage)?;
            assert!(file_eq_assert(&mut mem_file, &mut new_mem_file)?);
            std::fs::rename(
                PathBuf::from("temp_transpose_file.md"),
                PathBuf::from("in_memory.md"),
            )?;
        }
        mem_file
    } else {
        // dummy case for type checking; flag allocation should prevent the
        // correctness assert from happening if this path is taken
        File::open(&target_file)?
    };

    // memmap solution
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
                "{style_bold}On average it took {:?}",
                total_duration / cli.times as u32
            );
        }
        print_throughput(size * cli.times as u64, total_duration);
        print!("{color_reset}{style_reset}\n");
        if cli.verbose {
            println!("mmap file looks like this:");
            sample_file(dims, &mut mmap_file)?;
        }
        if cli.check_work && cli.in_memory {
            assert!(file_eq_assert(&mut mem_file, &mut mmap_file)?);
        }
    }

    // naive entirely on disk
    if cli.on_disk {
        print!("{color_bright_blue}");
        println!("starting transpose entirely on disk");
        #[cfg(unix)]
        {
            let mut total_duration = Duration::from_secs(0);
            let mut on_disk_result_file = File::open(PathBuf::from("input_file.md"))?;
            for _ in 0..cli.times {
                let (new_joined_file, disk_dur) = disk_io_solution(dims, &target_file)?;
                total_duration += disk_dur;
                on_disk_result_file = new_joined_file;
            }
            if cli.times > 1 {
                println!(
                    "{style_bold}On average it took {:?}",
                    total_duration / cli.times as u32
                );
            }
            print_throughput(size * cli.times as u64, total_duration);
            print!("{color_reset}{style_reset}\n");
            if cli.verbose {
                println!("disk manipulated file looks like this:");
                sample_file(dims, &mut on_disk_result_file)?;
            }
            if cli.check_work && cli.in_memory {
                assert!(file_eq_assert(&mut mem_file, &mut on_disk_result_file)?);
            }
        }
        #[cfg(not(unix))]
        println!("function not available on non-unix systems")
    }

    // buffered but entirely on disk
    if cli.buff_on_disk {
        print!("{color_bright_green}");
        println!("starting transpose on disk but buffered");
        #[cfg(unix)]
        {
            let mut total_duration = Duration::from_secs(0);
            let mut buffered_disk_result_file = File::open(PathBuf::from("input_file.md"))?;
            for _ in 0..cli.times {
                let (new_joined_file, buff_disk_dur) =
                    buffered_disk_io_solution(dims, &target_file)?;
                total_duration += buff_disk_dur;
                buffered_disk_result_file = new_joined_file;
            }
            if cli.times > 1 {
                println!(
                    "{style_bold}On average it took {:?}",
                    total_duration / cli.times as u32
                );
            }
            print_throughput(size * cli.times as u64, total_duration);
            print!("{color_reset}{style_reset}\n");
            if cli.verbose {
                println!("buffered disk manipulated file looks like this:");
                sample_file(dims, &mut buffered_disk_result_file)?;
            }
            if cli.check_work && cli.in_memory {
                assert!(file_eq_assert(
                    &mut mem_file,
                    &mut buffered_disk_result_file
                )?);
            }
        }
        #[cfg(not(unix))]
        println!("function not available on non-unix systems")
    }

    // multiple temp files concatenated solution
    if cli.join {
        print!("{color_cyan}");
        println!("starting transpose with temp files");
        let mut total_duration = Duration::from_secs(0);
        let mut joined_file = File::open(PathBuf::from("input_file.md"))?;
        for _ in 0..cli.times {
            let (new_joined_file, temp_dur) = join_file_handles(dims, &mut input_handle)?;
            total_duration += temp_dur;
            joined_file = new_joined_file;
        }
        if cli.times > 1 {
            println!(
                "{style_bold}On average it took {:?}",
                total_duration / cli.times as u32
            );
        }
        print_throughput(size * cli.times as u64, total_duration);
        print!("{color_reset}{style_reset}\n");
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

fn setup_file(Dimensions { size, .. }: Dimensions, target_file: &Path) -> Result<File> {
    let handle = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&target_file)?;
    if target_file.metadata()?.len() != size {
        println!("setting up file to work on");
        handle.set_len(size)?;
        let mut buffered_writer = BufWriter::new(&handle);
        let letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$^&*()-+[]"
            .to_string();
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
    let mut output_buff = vec![0; size as usize];
    // transpose the data in memory
    for i in 0..rows {
        for j in 0..cols {
            output_buff[j * rows + i] = input_buff[i * cols + j];
        }
    }
    assert_eq!(size as usize, output_file.write(&output_buff)?);
    output_file.flush()?;
    output_file.sync_all()?;
    let duration = start_time.elapsed();

    println!("in_memory time: {:?}", duration);

    Ok((output_file, duration))
}

fn mmap_solution(
    Dimensions { rows, cols, size }: Dimensions,
    input_path: &Path,
) -> Result<(File, Duration)> {
    let input_file = File::open(input_path)?;
    let target_path: PathBuf = PathBuf::from("mmap.md");
    // std::fs::copy(input_path, &target_path)?;
    let output_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&target_path)?;
    output_file.set_len(size)?;
    let input_mmap = unsafe { Mmap::map(&input_file)? };
    let mut output_mmap = unsafe { MmapMut::map_mut(&output_file)? };

    let start_time = Instant::now();
    for i in 0..rows {
        for j in 0..cols {
            output_mmap[j * rows + i] = input_mmap[i * cols + j];
        }
    }

    output_mmap.flush()?;
    output_file.sync_all()?;

    let duration = start_time.elapsed();
    println!("memmap time: {:?}", duration);

    Ok((output_file, duration))
}

#[cfg(unix)]
fn disk_io_solution(
    Dimensions { rows, cols, size }: Dimensions,
    input_path: &Path,
) -> Result<(File, Duration)> {
    let input_file = File::open(&input_path)?;
    let target_path: PathBuf = PathBuf::from("disk_io.md");
    let mut output_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&target_path)?;
    output_file.set_len(size)?;

    let start_time = Instant::now();
    let mut input_row_buf = vec![0u8; cols];
    for i in 0..rows {
        input_file.read_at(&mut input_row_buf, (i * cols) as u64)?;
        for j in 0..cols {
            output_file.write_at(&[input_row_buf[j]], (j * rows + i) as u64)?;
        }
    }

    output_file.flush()?;
    output_file.sync_all()?;

    let duration = start_time.elapsed();
    println!("on_disk naive time: {:?}", duration);

    Ok((output_file, duration))
}

#[cfg(unix)]
fn buffered_disk_io_solution(
    Dimensions { rows, cols, size }: Dimensions,
    input_path: &Path,
) -> Result<(File, Duration)> {
    const BUFF_SIZE: usize = 2usize.pow(10);
    let input_file = File::open(&input_path)?;
    let mut input_file_reader = BufReader::with_capacity(BUFF_SIZE * 30, input_file);
    let target_path: PathBuf = PathBuf::from("disk_io.md");
    let mut output_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&target_path)?;
    output_file.set_len(size)?;

    let start_time = Instant::now();

    let mut output_buff_buff: Vec<Vec<u8>> = vec![Vec::with_capacity(BUFF_SIZE); cols];
    let mut input_row_buff = vec![0; cols];
    let mut write_index = 0;

    for row_index in 0..rows {
        input_file_reader.read(&mut input_row_buff)?;
        (&mut output_buff_buff, &input_row_buff)
            .into_par_iter()
            .for_each(|(row_buf, row_entry)| row_buf.push(*row_entry));
        if output_buff_buff.first().unwrap().len() >= BUFF_SIZE || row_index == rows - 1 {
            output_buff_buff
                .iter()
                .enumerate()
                .try_for_each(|(column_index, col_buf)| {
                    output_file
                        .write_at(col_buf, (write_index + column_index * rows) as u64)
                        .and_then(|_| Ok(()))
                })?;
            output_buff_buff
                .par_iter_mut()
                .for_each(|row_buf| row_buf.clear());
            write_index = row_index + 1;
        }
    }
    output_file.flush()?;
    output_file.sync_all()?;

    let duration = start_time.elapsed();
    println!("on_disk buffered time: {:?}", duration);

    Ok((output_file, duration))
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
    let io_result = || -> Result<(File, Duration)> {
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
            .collect::<Result<Vec<(PathBuf, BufWriter<File>)>>>()?;

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

        new_row_file_handles
            .into_iter()
            .map(|(handle, mut writer)| {
                writer.flush()?;
                std::io::copy(&mut File::open(&handle)?, &mut output_file)?;
                Ok(handle)
            })
            .collect::<Result<Vec<PathBuf>>>()?;
        output_file.flush()?;
        output_file.sync_all()?;
        let duration = start_time.elapsed();
        Ok((output_file, duration))
    }();

    let delete_result = || -> Result<_> {
        (0..rows)
            .map(|i| {
                let temp_file_name = temp_dir.join(format!("row-{}.md", i));
                if temp_file_name.exists() {
                    std::fs::remove_file(&temp_file_name)?
                }
                Ok(())
            })
            .fold(anyhow::Ok(()), |acc, res| acc.and(res))
    }();

    let (output_file, duration) = delete_result.and(io_result)?;
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

fn print_throughput(bytes_processed: u64, total_duration: Duration) {
    let throughput = (bytes_processed as f64 / total_duration.as_secs_f64()).floor() as usize;
    println!("Average throughput {}/s", Size::from_bytes(throughput));
}

#[unsafe(no_mangle)]
fn calculate_index(i: usize, len: usize) -> usize {
    (i * 257) % len
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_all() {
        let cli = Cli {
            log2_size: 5,
            verbose: true,
            check_work: true,
            times: 3,
            in_memory: true,
            mmap: true,
            join: true,
            on_disk: true,
            buff_on_disk: true,
            all: false,
        };
        _main(cli).unwrap();
    }
}
