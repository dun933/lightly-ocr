package db

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	log "github.com/sirupsen/logrus"

	// mysql driver
	_ "github.com/go-sql-driver/mysql"
)

//constant
const userQuery string = "CREATE TABLE IF NOT EXISTS database (userName varchar(255), userScore int, imgPath varchar(255));"
const co2Query string = "CREATE TABLE IF NOT EXISTS co2 (items varchar(255), emission float64);"
const dbName string = "backend-app"
const devEnv = false

// insertQuery holds default string to insert item into database
var insertQuery = fmt.Sprintf("INSERT INTO %s VALUES (?,?,?)", dbName)

var db *sql.DB
var connected bool

// cnnstr = fmt.Sprintf("root:toor@tcp(localhost)/%s", dbName) -> devEnv

// cnnstr = fmt.Sprintf("application:application123@tcp(localhost)/%s", dbName)

// ErrNotConnected should be thrown when there is no established connection to the database
var ErrNotConnected error = errors.New("No connection to database")

// PingTimeout measures how long we should ping when attempting to reconnect to the database
var PingTimeout time.Duration = 1 * time.Second

// SleepTimeout returns the amount of time to wait during the goroutine when reconnecting
var SleepTimeout time.Duration = 5 * time.Second

// User contains the general table to store username and given score for the item
type User struct {
	userName  string
	userScore float64
	imgPath   string
}

// CO2 contains data about item with agrigation sum of greenhouse gasses (CO2, NH4, etc)
type CO2 struct {
	items    string
	emission float64
}

type DB struct {
	Client    *sql.DB
	Driver    string
	URL       string
	connected bool
}

// FetchUser returns a query in backend-app
func FetchUser() ([]User, error) {
	if connected {
		selectQuery := fmt.Sprintf("SELECT * FROM %s", dbName)
		row, err := db.Query(selectQuery)
		defer row.Close()

		user := []User{}

		for row.Next() {
			u := User{}
			if err := row.Scan(&u.userName, &u.userScore, &u.imgPath); err != nil {
				log.Fatal(err)
			} else {
				user = append(user, u)
			}
		}
		return user, err
	}
	return nil, ErrNotConnected
}

// init tries to reconnect to database when there is no connection established
func init() {
	connected = false
	db = connectDB()

	err := createTable(userQuery)
	if err != nil {
		panic(err)
	}
	go func() {
		for {
			ctx, cancel := context.WithTimeout(context.Background(), PingTimeout)
			defer cancel()
			err := db.PingContext(ctx)
			if err != nil {
				connected = false
				log.Errorf("attempting to reconnect, connection: %s", err.Error())
				db = connectDB()
			} else {
				connected = true
			}
			time.Sleep(SleepTimeout)
		}
	}()

}
